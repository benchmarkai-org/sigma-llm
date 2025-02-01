from flask import Flask, request, jsonify, abort
import time
import os
from dotenv import load_dotenv
from functools import wraps
import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from secrets import compare_digest
from pathlib import Path
from sigma_llm.llm import LLMManager

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": os.getenv("ALLOWED_ORIGINS", "").split(",")}})

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per day", "10 per minute"]
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

SERVICE_API_KEY = os.getenv('SERVICE_API_KEY')
llm_manager = LLMManager()

# Secure headers middleware
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

def _validate_api_key():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        logger.warning("Unauthorized access attempt: Missing or invalid authorization header")
        abort(401, description='Missing or invalid authorization header')
    
    api_key = auth_header.split(' ')[1]
    if not compare_digest(api_key, SERVICE_API_KEY):
        logger.warning("Unauthorized access attempt: Invalid API key")
        abort(401, description='Invalid API key')

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        _validate_api_key()
        return f(*args, **kwargs)
    return decorated

@app.route('/api/v1/rules', methods=['POST'])
@limiter.limit("10 per minute")
@require_api_key
def create_rule():
    try:
        data = request.get_json()
        if not data:
            logger.error("Bad request: Missing request body")
            abort(400, description='Missing request body')
            
        query = data.get('query')
        if not query:
            logger.error("Bad request: Missing query field")
            abort(400, description='Missing query field')
        if len(query) > 1000:
            logger.error("Bad request: Query too long")
            abort(400, description='Query too long')
            
        yaml_block = llm_manager.generate_rule(query)
        if not yaml_block or len(yaml_block) > 10000:
            logger.error("Internal server error: Invalid rule generated")
            abort(500, description='Invalid rule generated')
            
        return jsonify({'rule': yaml_block})
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        abort(500, description='Internal server error')

@app.route('/api/v1/judge', methods=['POST'])
@limiter.limit("10 per minute")
@require_api_key
def judge_rules():
    try:
        data = request.get_json()
        if not data:
            logger.error("Bad request: Missing request body")
            abort(400, description='Missing request body')
            
        rule1 = data.get('rule1')
        rule2 = data.get('rule2')
        if not all([rule1, rule2]):
            logger.error("Bad request: Missing required fields")
            abort(400, description='Missing required fields')
        if len(rule1) > 5000 or len(rule2) > 5000:
            logger.error("Bad request: Rules too long")
            abort(400, description='Rules too long')
            
        judgment = llm_manager.judge_rules(rule1, rule2)
        return jsonify({"judgment": judgment})
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        abort(500, description='Internal server error')

@app.route('/api/v1/assess', methods=['POST'])
@limiter.limit("10 per minute")
@require_api_key
def assess_rule():
    try:
        data = request.get_json()
        if not data:
            logger.error("Bad request: Missing request body")
            abort(400, description='Missing request body')
            
        rule = data.get('rule')
        if not rule:
            logger.error("Bad request: Missing rule field")
            abort(400, description='Missing rule field')
        if len(rule) > 5000:
            logger.error("Bad request: Rule too long")
            abort(400, description='Rule too long')
            
        assessment = llm_manager.assess_rule(rule)
        return jsonify({"assessment": assessment})
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        abort(500, description='Internal server error')

@app.route('/api/v1/summarize-references', methods=['POST'])
@limiter.limit("5 per minute")
@require_api_key
def summarize_references():
    try:
        data = request.get_json()
        if not data:
            logger.error("Bad request: Missing request body")
            abort(400, description='Missing request body')
            
        reference_content = data.get('reference_content')
        if not reference_content:
            logger.error("Bad request: Missing reference_content field")
            abort(400, description='Missing reference_content field')
            
        summary = llm_manager.summarize_references(reference_content)
        return jsonify({'summary': summary})
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        abort(500, description='Internal server error')

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    load_dotenv()
    app.config['DEBUG'] = os.getenv('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=8080)