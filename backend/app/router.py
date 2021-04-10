from api.index import index_bp


def routers(app):
    app.register_blueprint(index_bp)

    return True


def csrf_exempt(csrf):

    return True
