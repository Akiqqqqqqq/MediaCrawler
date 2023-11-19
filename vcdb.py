import os
from pathlib import Path

from milvus import default_server
from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      connections, utility)

_URI = os.environ.get('MILVUS_URI', 'http://localhost:1953')
_TOKEN = os.environ.get('MILVUS_TOKEN')

# Const names
_COLLECTION_NAME = 'user_memory'

# Vector parameters
_DIM = 1536

# Index parameters
_METRIC_TYPE = 'L2'
_INDEX_TYPE = 'IVF_FLAT'
_NLIST = 2048
_NPROBE = 16
_TOPK = 3


class VCDB:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VCDB, cls).__new__(cls)
        return cls._instance

    def __init__(self, uri=None, token=None, collection=None, embed=False):
        # We need a way to determine if the actual __init__ code has been executed before
        if not hasattr(self, "initialized"):
            self.initialized = True

            if uri is None:
                uri = _URI
            if token is None:
                token = _TOKEN
            if embed:
                self.embed = True
                self.create_lite_server(uri=uri, token=token)
            else:
                self.embed = False
            self.create_connection(uri=uri, token=token)

            if collection is None:
                collection = _COLLECTION_NAME

            self.create_or_get_collection(collection)
            self.load_collection()
    
    def create_lite_server(self, uri, token):
        print('starting embed server...')
        home = Path.home()
        db_path = home / '.noiz' / 'mm' / 'vcdb'
        if not db_path.exists():
            os.makedirs(db_path.parent, exist_ok=True)
        default_server.start()
        connections.connect(uri=uri, token=token)
        print(utility.get_server_version())

    def create_connection(self, uri, token):
        print(f"\nCreate connection...")
        connections.connect(uri=uri, token=token)
        print("connected to milvus.")

    def insert(self, data):
        self.collection.insert(data)
        self.collection.flush()

    def upsert(self, data):
        self.collection.upsert(data)
        self.collection.flush()
    
    def delete(self, expr):
        self.collection.delete(expr=expr)
    
    def query(self, expr, limit, ouput_fields):
        return self.collection.query(
            expr=expr,
            limit=limit,
            output_fields=ouput_fields
        )

    def search(self, search_vectors, user_id, limit=_TOPK):
        search_param = {
            "data": search_vectors,
            "anns_field": 'content',
            "param": {"metric_type": _METRIC_TYPE, "params": {"nprobe": _NPROBE}},
            "limit": limit,
            "output_fields": ['content_str','weight','ref_count','time'],
            "expr": f"id >= 0 and user_id == '{user_id}'"}
        return self.collection.search(**search_param)
        
    def get_entity_num(self):
        return self.collection.num_entities

    def create_or_get_collection(self, name):
        if utility.has_collection(name):
            self.collection = Collection(name)      
            return
        print("creating collection...")
        field1 = FieldSchema(name='id', dtype=DataType.INT64, description="int64", is_primary=True, auto_id=True)
        field2 = FieldSchema(name='content', dtype=DataType.FLOAT_VECTOR, description="embedding vectors", dim=_DIM,
                             is_primary=False)
        field3 = FieldSchema(name='content_str', dtype=DataType.VARCHAR, description="content", max_length=65530 ,is_primary=False)
        field4 = FieldSchema(name='weight', dtype=DataType.DOUBLE, description="weight of memory", is_primary=False)
        field5 = FieldSchema(name='ref_count', dtype=DataType.INT64, description="ref count", is_primary=False)
        field6 = FieldSchema(name='time', dtype=DataType.VARCHAR, description="time of record", max_length=30, is_primary=False)
        field7 = FieldSchema(name='user_id', dtype=DataType.VARCHAR,
                             description="user id", max_length=65530, is_primary=False)
        schema = CollectionSchema(fields=[
                                  field1, field2, field3, field4, field5, field6, field7], description="memory about user")
        self.collection = Collection(name=name,schema=schema)
        self.create_index()
        print("\ncollection created:", name)


    # Drop a collection in Milvus
    def drop_collection(self):
        self.collection.drop()
        print("\nDrop collection: {}".format(self.collection.name))

    def create_index(self):
        print("creating index...")
        index_param = {
            "index_type": _INDEX_TYPE,
            "params": {"nlist": _NLIST},
            "metric_type": _METRIC_TYPE}
        self.collection.create_index(field_name='content', index_params=index_param)
        print("\nCreated index:\n{}".format(self.collection.index().params))


    def drop_index(self):
        self.collection.drop_index()
        print("\nDrop index sucessfully")


    def load_collection(self):
        self.collection.load()


    def release_collection(self):
        self.collection.release()
    
    def close(self):
        self.release_collection()
        if self.embed is True:
            default_server.stop()

