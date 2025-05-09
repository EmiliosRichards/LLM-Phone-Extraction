import unittest
from unittest.mock import patch, MagicMock
import threading
import logging
import os
from ..rate_limiter import RateLimiter

class TestRateLimiterConcurrency(unittest.TestCase):
    def setUp(self) -> None:
        """Initialize test environment with mocked time and rate limiter instance."""
        # Use a fixed time for consistent testing
        self.time_patcher = patch('time.time')
        self.mock_time = self.time_patcher.start()
        self.mock_time.return_value = 0.0
        
        # Create a rate limiter with known values
        self.rate_limiter = RateLimiter(
            requests_per_second=10.0,  # 10 requests per second
            burst_size=5,             # Allow burst of 5 requests
            resource_name="test"
        )
        
        # Configure logging for debug output
        logging.basicConfig(level=logging.DEBUG)
    
    def tearDown(self) -> None:
        """Clean up test environment by stopping time patcher."""
        self.time_patcher.stop()
    
    @unittest.skipUnless(os.getenv("ENABLE_CONCURRENCY_TESTS"), "Concurrency tests disabled in CI")
    def test_thread_safety(self) -> None:
        """Should maintain correct token count when multiple threads acquire tokens simultaneously."""
        def acquire_tokens(limiter, results, index, barrier):
            # Wait for all threads to be ready
            barrier.wait()
            results[index] = limiter.acquire()
            logging.debug(f"Thread {index} acquired: {results[index]}")
        
        # Create multiple threads trying to acquire tokens simultaneously
        results = [None] * 5
        threads = []
        barrier = threading.Barrier(5)  # Barrier for 5 threads
        for i in range(5):
            thread = threading.Thread(
                target=acquire_tokens,
                args=(self.rate_limiter, results, i, barrier)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify that exactly 5 tokens were acquired
        self.assertEqual(sum(1 for r in results if r), 5)
        self.assertEqual(self.rate_limiter.tokens, 0.0)

    @unittest.skipUnless(os.getenv("ENABLE_CONCURRENCY_TESTS"), "Concurrency tests disabled in CI")
    def test_thread_contention(self) -> None:
        """Should handle contention when more threads request tokens than are available."""
        def acquire_tokens(limiter, results, index, barrier):
            # Wait for all threads to be ready
            barrier.wait()
            results[index] = limiter.acquire()
            logging.debug(f"Thread {index} acquired: {results[index]}")
        
        # Create 10 threads trying to acquire tokens (but only 5 are available)
        results = [None] * 10
        threads = []
        barrier = threading.Barrier(10)  # Barrier for 10 threads
        for i in range(10):
            thread = threading.Thread(
                target=acquire_tokens,
                args=(self.rate_limiter, results, i, barrier)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify that exactly 5 tokens were acquired (matching burst_size)
        successful_acquires = sum(1 for r in results if r)
        self.assertEqual(successful_acquires, 5)
        self.assertEqual(self.rate_limiter.tokens, 0.0)

    @unittest.skipUnless(os.getenv("ENABLE_CONCURRENCY_TESTS"), "Concurrency tests disabled in CI")
    def test_timeout_under_contention(self) -> None:
        """Should handle timeouts correctly when multiple threads wait for tokens."""
        def acquire_with_timeout(limiter, results, index, barrier):
            # Wait for all threads to be ready
            barrier.wait()
            results[index] = limiter.acquire(timeout=0.1)
            logging.debug(f"Thread {index} acquired: {results[index]}")
        
        # First, deplete all available tokens
        for _ in range(5):
            self.rate_limiter.acquire()
        
        # Now create 5 threads that will try to acquire tokens with timeout
        results = [None] * 5
        threads = []
        barrier = threading.Barrier(5)  # Barrier for 5 threads
        for i in range(5):
            thread = threading.Thread(
                target=acquire_with_timeout,
                args=(self.rate_limiter, results, i, barrier)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify that no tokens were acquired (all should timeout)
        successful_acquires = sum(1 for r in results if r)
        self.assertEqual(successful_acquires, 0)
        self.assertEqual(self.rate_limiter.tokens, 0.0)

if __name__ == '__main__':
    unittest.main() 