# This file is modified from DeepDB author's
from Schemas.graph_representation import SchemaGraph, Table


def gen_stack_schema(csv_path):
    """
    Specifies full stack schema.
    """
    schema = SchemaGraph()

    # tables

    # account
    # id contains -1
    schema.add_table(Table('account', attributes=['id', 'display_name', 'location', 'about_me', 'website_url'],
                           csv_file_location=csv_path.format('account'),
                           table_size=13872153))

    # answer
    schema.add_table(Table('answer', attributes=['id', 'site_id', 'question_id', 'creation_date', 'deletion_date', 'score', 'view_count', 'body', 'owner_user_id', 'last_editor_id', 'last_edit_date', 'last_activity_date', 'title'],
                           csv_file_location=csv_path.format('answer'),
                           table_size=6347553))

    # question
    # owner_user_id contains -1
    schema.add_table(Table('question', attributes=['id', 'site_id', 'accepted_answer_id', 'creation_date', 'deletion_date', 'score', 'view_count', 'body', 'owner_user_id', 'last_editor_id', 'last_edit_date', 'last_activity_date', 'title', 'favorite_count', 'closed_date', 'tagstring'],
                           csv_file_location=csv_path.format('question'),
                           table_size=12666441))

    # site
    schema.add_table(Table('site', attributes=['site_id', 'site_name'],
                           csv_file_location=csv_path.format('site'),
                           table_size=173))

    # so_user
    # id contains -1
    schema.add_table(Table('so_user', attributes=['id', 'site_id', 'reputation', 'creation_date', 'last_access_date', 'upvotes', 'downvotes', 'account_id'],
                           csv_file_location=csv_path.format('so_user'),
                           table_size=21097302))

    # tag
    schema.add_table(Table('tag', attributes=['id', 'site_id', 'name'],
                           csv_file_location=csv_path.format('tag'),
                           table_size=186770))

    # tag_question
    schema.add_table(Table('tag_question', attributes=['question_id', 'tag_id', 'site_id'],
                           csv_file_location=csv_path.format('tag_question'),
                           table_size=36883819))


    # ---------- relationship addition ------------

    schema.add_relationship('tag', 'site_id', 'site', 'site_id')
    schema.add_relationship('question', 'site_id', 'site', 'site_id')
    schema.add_relationship('tag_question', 'site_id', 'site', 'site_id')

    schema.add_relationship('tag_question', 'question_id', 'question', 'id')
    schema.add_relationship('tag_question', 'tag_id', 'tag', 'id')

    schema.add_relationship('answer', 'question_id', 'question', 'id')
    schema.add_relationship('answer', 'owner_user_id', 'so_user', 'id')
    schema.add_relationship('so_user', 'account_id', 'account', 'id')
    schema.add_relationship('question', 'owner_user_id', 'so_user', 'id')

    schema.add_relationship('answer', 'site_id', 'question', 'site_id')
    schema.add_relationship('answer', 'site_id', 'so_user', 'site_id')
    schema.add_relationship('question', 'site_id', 'so_user', 'site_id')

    return schema
