import logging
import queue
import time
from datetime import datetime, timezone
from flask import Blueprint, Response, request

from app.services.websocket_service import get_realtime_service, sse_publisher

logger = logging.getLogger(__name__)

ws_bp = Blueprint('websocket', __name__)


@ws_bp.route('/stream/<channel>')
def sse_stream(channel):
    """Server-Sent Events stream for real-time updates
    
    Channels: market, price, signal, all
    """
    q = queue.Queue(maxsize=10)
    sse_publisher.add_connection(channel, q)
    
    def generate():
        try:
            while True:
                try:
                    message = q.get(timeout=30)
                    yield message
                except queue.Empty:
                    yield f"data: {{'ping': true}}\n\n"
        except GeneratorExit:
            sse_publisher.remove_connection(channel, q)
    
    return Response(generate(), mimetype='text/event-stream')


@ws_bp.route('/stream/status')
def stream_status():
    """Get streaming service status"""
    svc = get_realtime_service()
    return svc.get_status()


@ws_bp.route('/stream/start', methods=['POST'])
def start_stream():
    """Start real-time polling"""
    interval = request.json.get('interval', 5) if request.json else 5
    svc = get_realtime_service()
    svc.start_polling(interval)
    return {'status': 'started', 'interval': interval}


@ws_bp.route('/stream/stop', methods=['POST'])
def stop_stream():
    """Stop real-time polling"""
    svc = get_realtime_service()
    svc.stop_polling()
    return {'status': 'stopped'}