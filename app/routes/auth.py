import logging
from flask import Blueprint, jsonify, request
from datetime import datetime
from functools import wraps

from app.models.db import db, User, StrategyProfile

logger = logging.getLogger(__name__)

auth_bp = Blueprint('auth', __name__)


def get_current_user():
    """Get current user from API key or session"""
    api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
    if api_key:
        user = User.query.filter_by(api_key=api_key, is_active=True).first()
        if user:
            return user
    
    return None


def get_user_or_guest():
    """Get current user or None for guest mode"""
    return get_current_user()


def require_auth(f):
    """Decorator requiring authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated


def optional_auth(f):
    """Decorator allowing guest mode with warning"""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if user:
            kwargs['current_user'] = user
        else:
            kwargs['current_user'] = None
            request.guest_mode_warning = True
        return f(*args, **kwargs)
    return decorated


def require_admin(f):
    """Decorator requiring admin auth"""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user or not user.is_admin:
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated


@auth_bp.route('/auth/register', methods=['POST'])
def register():
    """Register new user
    
    Body:
        username: str
        email: str
        password: str
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not username or not email or not password:
            return jsonify({'error': 'username, email, password required'}), 400
        
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        user = User(username=username, email=email)
        user.set_password(password)
        user.generate_api_key()
        
        db.session.add(user)
        db.session.commit()
        
        default_profile = StrategyProfile(
            user_id=user.id,
            name='Default',
            description='Default strategy profile',
            is_active=True
        )
        db.session.add(default_profile)
        db.session.commit()
        
        logger.info(f'User registered: {username}')
        
        return jsonify({
            'message': 'User registered successfully',
            'user': user.to_dict(include_private=True)
        }), 201
    
    except Exception as e:
        logger.error(f'Registration error: {e}')
        return jsonify({'error': str(e)}), 500


@auth_bp.route('/auth/login', methods=['POST'])
def login():
    """Login user
    
    Body:
        username: str
        password: str
    
    Returns:
        api_key for authenticated requests
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'error': 'username and password required'}), 400
        
        user = User.query.filter_by(username=username).first()
        
        if not user or not user.verify_password(password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        if not user.is_active:
            return jsonify({'error': 'Account disabled'}), 403
        
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        if not user.api_key:
            user.generate_api_key()
            db.session.commit()
        
        logger.info(f'User logged in: {username}')
        
        return jsonify({
            'message': 'Login successful',
            'user': user.to_dict(),
            'api_key': user.api_key
        })
    
    except Exception as e:
        logger.error(f'Login error: {e}')
        return jsonify({'error': str(e)}), 500


@auth_bp.route('/auth/logout', methods=['POST'])
@require_auth
def logout():
    """Logout current user"""
    return jsonify({'message': 'Logged out successfully'})


@auth_bp.route('/auth/me', methods=['GET'])
@require_auth
def get_me():
    """Get current user info"""
    user = get_current_user()
    return jsonify({'user': user.to_dict(include_private=True)})


@auth_bp.route('/auth/api-key', methods=['POST'])
@require_auth
def regenerate_api_key():
    """Regenerate API key"""
    user = get_current_user()
    old_key = user.api_key
    new_key = user.generate_api_key()
    db.session.commit()
    
    logger.info(f'API key regenerated for user: {user.username}')
    
    return jsonify({
        'message': 'API key regenerated',
        'api_key': new_key
    })


@auth_bp.route('/auth/profiles', methods=['GET'])
@require_auth
def list_profiles():
    """List user's strategy profiles"""
    user = get_current_user()
    profiles = StrategyProfile.query.filter_by(user_id=user.id).all()
    
    return jsonify({
        'profiles': [p.to_dict() for p in profiles]
    })


@auth_bp.route('/auth/profiles', methods=['POST'])
@require_auth
def create_profile():
    """Create new strategy profile
    
    Body:
        name: str
        description: str (optional)
        thresholds: dict (optional)
        filters: dict (optional)
        regimes: dict (optional)
        timing: dict (optional)
        alerts: dict (optional)
    """
    try:
        user = get_current_user()
        data = request.get_json() or {}
        
        name = data.get('name')
        if not name:
            return jsonify({'error': 'name required'}), 400
        
        profile = StrategyProfile(
            user_id=user.id,
            name=name,
            description=data.get('description', '')
        )
        
        if 'thresholds' in data:
            profile.min_confidence_weak = data['thresholds'].get('min_confidence_weak', 50)
            profile.min_confidence_moderate = data['thresholds'].get('min_confidence_moderate', 60)
            profile.min_confidence_strong = data['thresholds'].get('min_confidence_strong', 75)
            profile.min_confidence_elite = data['thresholds'].get('min_confidence_elite', 90)
        
        if 'filters' in data:
            profile.max_spread = data['filters'].get('max_spread', 2.0)
            profile.min_volume = data['filters'].get('min_volume', 5000)
            profile.max_orderbook_imbalance = data['filters'].get('max_orderbook_imbalance', 80)
        
        if 'regimes' in data:
            profile.allow_trending = data['regimes'].get('allow_trending', True)
            profile.allow_sideways = data['regimes'].get('allow_sideways', True)
            profile.allow_breakout = data['regimes'].get('allow_breakout', True)
            profile.allow_whipsaw = data['regimes'].get('allow_whipsaw', False)
        
        if 'timing' in data:
            profile.max_time_remaining = data['timing'].get('max_time_remaining', 300)
            profile.early_exit_enabled = data['timing'].get('early_exit_enabled', False)
            profile.early_exit_threshold = data['timing'].get('early_exit_threshold', 0.80)
        
        if 'alerts' in data:
            profile.alerts_enabled = data['alerts'].get('enabled', True)
            profile.alert_telegram = data['alerts'].get('telegram', True)
            profile.alert_discord = data['alerts'].get('discord', False)
            profile.alert_email = data['alerts'].get('email', False)
        
        db.session.add(profile)
        db.session.commit()
        
        logger.info(f'Profile created: {profile.name} for user {user.username}')
        
        return jsonify({
            'message': 'Profile created',
            'profile': profile.to_dict()
        }), 201
    
    except Exception as e:
        logger.error(f'Profile creation error: {e}')
        return jsonify({'error': str(e)}), 500


@auth_bp.route('/auth/profiles/<profile_id>', methods=['GET'])
@require_auth
def get_profile(profile_id):
    """Get specific profile"""
    user = get_current_user()
    profile = StrategyProfile.query.filter_by(
        profile_id=profile_id,
        user_id=user.id
    ).first()
    
    if not profile:
        return jsonify({'error': 'Profile not found'}), 404
    
    return jsonify({'profile': profile.to_dict()})


@auth_bp.route('/auth/profiles/<profile_id>', methods=['PUT'])
@require_auth
def update_profile(profile_id):
    """Update strategy profile"""
    try:
        user = get_current_user()
        profile = StrategyProfile.query.filter_by(
            profile_id=profile_id,
            user_id=user.id
        ).first()
        
        if not profile:
            return jsonify({'error': 'Profile not found'}), 404
        
        data = request.get_json() or {}
        
        if 'name' in data:
            profile.name = data['name']
        if 'description' in data:
            profile.description = data['description']
        if 'is_active' in data:
            profile.is_active = data['is_active']
        
        if 'thresholds' in data:
            profile.min_confidence_weak = data['thresholds'].get('min_confidence_weak', profile.min_confidence_weak)
            profile.min_confidence_moderate = data['thresholds'].get('min_confidence_moderate', profile.min_confidence_moderate)
            profile.min_confidence_strong = data['thresholds'].get('min_confidence_strong', profile.min_confidence_strong)
            profile.min_confidence_elite = data['thresholds'].get('min_confidence_elite', profile.min_confidence_elite)
        
        if 'filters' in data:
            profile.max_spread = data['filters'].get('max_spread', profile.max_spread)
            profile.min_volume = data['filters'].get('min_volume', profile.min_volume)
            profile.max_orderbook_imbalance = data['filters'].get('max_orderbook_imbalance', profile.max_orderbook_imbalance)
        
        if 'regimes' in data:
            profile.allow_trending = data['regimes'].get('allow_trending', profile.allow_trending)
            profile.allow_sideways = data['regimes'].get('allow_sideways', profile.allow_sideways)
            profile.allow_breakout = data['regimes'].get('allow_breakout', profile.allow_breakout)
            profile.allow_whipsaw = data['regimes'].get('allow_whipsaw', profile.allow_whipsaw)
        
        if 'timing' in data:
            profile.max_time_remaining = data['timing'].get('max_time_remaining', profile.max_time_remaining)
            profile.early_exit_enabled = data['timing'].get('early_exit_enabled', profile.early_exit_enabled)
            profile.early_exit_threshold = data['timing'].get('early_exit_threshold', profile.early_exit_threshold)
        
        if 'alerts' in data:
            profile.alerts_enabled = data['alerts'].get('enabled', profile.alerts_enabled)
            profile.alert_telegram = data['alerts'].get('telegram', profile.alert_telegram)
            profile.alert_discord = data['alerts'].get('discord', profile.alert_discord)
            profile.alert_email = data['alerts'].get('email', profile.alert_email)
        
        db.session.commit()
        
        logger.info(f'Profile updated: {profile.name}')
        
        return jsonify({
            'message': 'Profile updated',
            'profile': profile.to_dict()
        })
    
    except Exception as e:
        logger.error(f'Profile update error: {e}')
        return jsonify({'error': str(e)}), 500


@auth_bp.route('/auth/profiles/<profile_id>', methods=['DELETE'])
@require_auth
def delete_profile(profile_id):
    """Delete strategy profile"""
    user = get_current_user()
    profile = StrategyProfile.query.filter_by(
        profile_id=profile_id,
        user_id=user.id
    ).first()
    
    if not profile:
        return jsonify({'error': 'Profile not found'}), 404
    
    profile_count = StrategyProfile.query.filter_by(user_id=user.id).count()
    if profile_count <= 1:
        return jsonify({'error': 'Cannot delete last profile'}), 400
    
    db.session.delete(profile)
    db.session.commit()
    
    logger.info(f'Profile deleted: {profile_id}')
    
    return jsonify({'message': 'Profile deleted'})


@auth_bp.route('/auth/profiles/<profile_id>/activate', methods=['POST'])
@require_auth
def activate_profile(profile_id):
    """Activate a profile (deactivates others)"""
    user = get_current_user()
    profile = StrategyProfile.query.filter_by(
        profile_id=profile_id,
        user_id=user.id
    ).first()
    
    if not profile:
        return jsonify({'error': 'Profile not found'}), 404
    
    StrategyProfile.query.filter_by(user_id=user.id).update({'is_active': False})
    profile.is_active = True
    db.session.commit()
    
    logger.info(f'Profile activated: {profile.name}')
    
    return jsonify({
        'message': 'Profile activated',
        'profile': profile.to_dict()
    })