from personal_area.auth import auth_bp
from personal_area.logout import logout_bp
from personal_area.registration import registration_bp
from personal_area.personal_area import personal_area_bp

from api.index import index_bp


def routers(app):
    app.register_blueprint(auth_bp)
    app.register_blueprint(logout_bp)
    app.register_blueprint(registration_bp)
    app.register_blueprint(personal_area_bp)
    app.register_blueprint(index_bp)

    return True


def csrf_exempt(csrf):
    csrf.exempt(auth_bp)
    csrf.exempt(logout_bp)
    csrf.exempt(registration_bp)
    csrf.exempt(personal_area_bp)
    csrf.exempt(index_bp)

    return True
