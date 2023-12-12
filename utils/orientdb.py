import pyorient
from decouple import config
import logging

class OrientDBClient:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.host = config('ORIENTDB_HOST')
        self.port = int(config('ORIENTDB_PORT'))
        self.username = config('ORIENTDB_USERNAME')
        self.password = config('ORIENTDB_PASSWORD')
        self.database = config('ORIENTDB_DB')
        self.client = None

    def connect(self):
        try:
            self.client = pyorient.OrientDB(self.host, self.port)
            self.client.connect(self.username, self.password)
            if not self.client.db_exists(self.database, pyorient.STORAGE_TYPE_PLOCAL):
                raise Exception(f"Database '{self.database}' does not exist.")
            self.client.db_open(self.database, self.username, self.password)
            self.logger.info(f"Connected to database '{self.database}'")
        except Exception as e:
            self.logger.error(f"Connection error: {e}")

    def execute_query(self, query):
        if self.client:
            try:
                result = self.client.command(query)
                return result
            except Exception as e:
                self.logger.info(f"Query execution error: {e}")
        else:
            self.logger.error("Not connected to the database.")

    def close(self):
        if self.client:
            self.client.db_close()
            self.logger.info(f"Disconnected from database '{self.database}'")