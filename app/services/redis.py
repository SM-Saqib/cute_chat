import redis
import json
from redis.exceptions import ConnectionError, RedisError


class RedisCache:
    """
    RedisCache class is a simple wrapper around the redis-py client.
    It provides a set_json method to store a JSON object in Redis,
    a get_json method to retrieve a JSON object from Redis,
    and a delete method to delete a key from Redis.
    """

    def __init__(self, host="localhost", port=6379, db=0, password=None):
        self.client = redis.StrictRedis(
            host=host, port=port, db=db, decode_responses=True, password=password
        )

    def set_json(self, key, data, ex=None):
        """
        Stores a JSON object in Redis.

        Args:
            key (str): Key to store the data under.
            data (dict or list): Data to store as JSON.
            ex (int, optional): Expiration time in seconds. Defaults to None.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        try:
            json_data = json.dumps(data)
            return self.client.set(key, json_data, ex=ex)
        except (ConnectionError, RedisError) as e:
            print(f"Redis error: {e}")
            return False

    def get_json(self, key):
        """
        Retrieves a JSON object from Redis.

        Args:
            key (str): Key to retrieve.

        Returns:
            dict or list: The retrieved JSON object, or None if the key doesn't exist or an error occurs.
        """
        try:
            json_data = self.client.get(key)
            if json_data:
                return json.loads(json_data)
            return None
        except (ConnectionError, RedisError, json.JSONDecodeError) as e:
            print(f"Redis error: {e}")
            return None

    def delete(self, key):
        """
        Deletes a key from Redis.

        Args:
            key (str): Key to delete.

        Returns:
            int: The number of keys that were removed.
        """
        try:
            return self.client.delete(key)
        except (ConnectionError, RedisError) as e:
            print(f"Redis error: {e}")
            return 0


# Example usage:
# cache = RedisCache()
# cache.set_json('my_key', {'name': 'John', 'age': 30})
# print(cache.get_json('my_key'))
# cache.delete('my_key')
