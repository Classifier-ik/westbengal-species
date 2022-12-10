from . import db
# from sqlalchemy import (BigInteger, Column, DateTime, ForeignKey,
#         Index, Integer, MetaData, SmallInteger, String, Table, Text)
from sqlalchemy import event, inspect, text
from sqlalchemy.dialects.mysql import TINYINT
from sqlalchemy.dialects.mysql.types import YEAR
# from sqlalchemy.ext.associationproxy import association_proxy

from sqlalchemy.sql import func
# from flask_login import UserMixin

class User(db.Model):
    __tablename__ = 'userstable'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.Text, nullable=False, unique=True)
    isAdmin = db.Column(TINYINT, server_default="0")
    password = db.Column(db.Text, nullable=False)


class Test(db.Model):
    __tablename__ = 'test'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.ForeignKey('userstable.id'),
        nullable=False, index=True
    )
    filepath = db.Column(db.Text, nullable=False, unique=True)
    predicted = db.Column(db.Text, nullable=False)
    userinput = db.Column(db.Text, nullable=False)
    validity = db.Column(db.Integer, nullable=True)

    user = db.relationship(
        'User',
        backref='test', lazy = True
    )

