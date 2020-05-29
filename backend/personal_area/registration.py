import re
import requests

from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash
from psycopg2 import sql

from app.models import check_auth, authorize, config
from pay import add_new_farmer_to_fns
from database import Database

registration_bp = Blueprint('registration', __name__)


@registration_bp.route('/registration', methods=['GET', 'POST'])
def registration():
    """Registration new user Page"""

    if request.headers.get('Token') != str(config['FLASK_APP']['SECRET_KEY']):
        return jsonify({'message': 'Не верный токен'}), 401, {'ContentType': 'application/json'}

    vozvrat = {}
    try:
        database = Database(config)
    except TypeError:
        vozvrat["messageError"] = "Нет подключения к БД"
        return jsonify(vozvrat)

    file = request.get_json(silent=True)
    if file != None and request.method == "POST":
        is_farmer = True if file["farmer"] == True else False
        user = {
            "email": file.get("email"),
            "password": file.get("password"),
            "confirm_password": file.get("confirm_password"),
            "firstname": file.get("firstname"),
            "lastname": file.get("lastname"),
            "patronymic": file.get("patronymic"),
            "number_phone": file.get("number_phone")
        }

        # Проверка введённых данных
        valid = valid_data(user, is_farmer)
        if valid != True:
            vozvrat["messageError"] = valid
            return jsonify(vozvrat)

        user['password'] = generate_password_hash(
            user['password'], method='sha256')
        result = execute_to_base(database, user)

        if result == True:
            valid = add_new_farmer_to_fns(user['inn'])
            if type(valid) == int:
                valid = execute_id(database, valid, user['email'])
            if valid != True:
                vozvrat["messageError"] = valid
                delete_user(database, user)
            else:
                valid = update_address(database, user)
                if valid != True:
                    vozvrat["messageError"] = valid
                    delete_user(database, user)
                else:
                    vozvrat["messageSuccess"] = "Пользователь зарегестрирован"
        else:
            vozvrat["messageError"] = result

    else:
        vozvrat["messageError"] = "JSON отсутсвует"
    return jsonify(vozvrat)


def execute_to_base(database, user):
    values_data = {}
    columns = {}
    values_data_adress = {}
    columns_address = {}
    for col in user:
        if not col in ["confirm_password"]:
            columns_address[col] = sql.Identifier(col)
            values_data_adress[col] = sql.Literal(user[col])

    query = sql.SQL("INSERT INTO {table}({column}) VALUES({value})").format(
        table=sql.Identifier("public", "users"),
        column=sql.SQL(',').join(
            columns[col] for col in columns),
        value=sql.SQL(',').join(
            values_data[val] for val in values_data),
    )
    vozvrat = database.insert_data(query)
    if vozvrat != True:
        return vozvrat

    return vozvrat


def valid_data(user, is_farmer):
    """Checking user data"""
    valid = valid_password(user['password'], user['confirm_password'])
    if valid != True:
        return valid
    valid = valid_email(user['email'])
    if valid != True:
        return valid
    return True


def valid_email(email):
    """Checking email"""
    if re.search(
            "[A-Z]{0,20}[a-z]{1,20}[\d]{0,20}[\.\-\@]{1,20}", email) == None:
        return "Почта не удовлетворяет требованиям"
    return True


def valid_password(password, confirm_password):
    """Checking password"""
    VALID_CHARS = [
        "[A-Z]{1,70}",
        "[a-z]{1,70}",
        "[0-9]{1,70}",
        "[\!\@\#\$\%\^\&\*\(\)\_\-\+\:\;\,\.]{0,70}"
    ]
    if password != confirm_password:
        return "Пароли не совпадают"
    for val in VALID_CHARS:
        if re.search(
                val, password) == None:
            return "Пароль не удовлетворяет требованиям"
    last_char = ""
    quantity = 0
    for char in password:
        if char != last_char:
            last_char = char
        elif char == last_char and quantity < 3:
            quantity += 1
        else:
            return "Пароль не удовлетворяет требованиям"
    return True
