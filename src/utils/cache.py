import math
import mmh3

class BloomFilter:
    __slots__ = ("capacity", "error_rate", "num_bits", "num_hashes", "bit_array")

    def __init__(self, capacity, error_rate):
        """
        Initializes a Bloom filter with a given capacity and error rate.

        Parameters::
        :param int capacity     : maximum number of items that can be stored in the filter
        :param float error_rate : desired false positive rate for the filter
        """
        self.capacity = capacity
        self.error_rate = error_rate

        if error_rate == 0:
            self.num_bits = 1
        else:
            self.num_bits =  math.ceil(-(capacity * math.log(self.error_rate)) / (math.log(2) ** 2))

        if self.num_bits == 0:
            self.num_bits = 1

        self.num_hashes = max(1, math.ceil((self.num_bits / self.capacity) * math.log(2)))
        self.bit_array = bytearray(math.ceil(self.num_bits / 8 ))

    def add(self, item):
        """
        Generate k hash indices for the given item using different seeds.

        Parameters:
            :param str item : item to add
        """
        if isinstance(item, bytes):
            item_bytes = item
        else:
            item_bytes = item.encode('utf-8')

        for i in range(self.num_hashes):
            digest = mmh3.hash(item_bytes, i) % self.num_bits

            byte_idx = digest // 8
            bit_offset = digest % 8
            self.bit_array[byte_idx] |= (1 << bit_offset)

    def __contains__(self, item):
        """
        Check if an item (bytes) is possibly in the Bloom filter.

        Parameters:
            :param str item : item to check
        """
        if isinstance(item, bytes):
            item_bytes = item
        else:
            item_bytes = item.encode('utf-8')

        for i in range(self.num_hashes):
            hash_val = mmh3.hash(item_bytes, i) % self.num_bits

            byte_idx = hash_val // 8
            bit_offset = hash_val % 8
            if not ((self.bit_array[byte_idx] >> bit_offset) & 1 == 1):
                return False
        return True