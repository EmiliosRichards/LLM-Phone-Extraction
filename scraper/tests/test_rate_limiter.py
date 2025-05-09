import unittest
from unittest.mock import patch, MagicMock
import time
from ..rate_limiter import RateLimiter
import os

class TestRateLimiter(unittest.TestCase):
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
    
    def tearDown(self) -> None:
        """Clean up test environment by stopping time patcher."""
        self.time_patcher.stop()
    
    def test_initial_state(self) -> None:
        """Should initialize rate limiter with correct default values."""
        self.assertEqual(self.rate_limiter.rate, 10.0)
        self.assertEqual(self.rate_limiter.capacity, 5.0)
        self.assertEqual(self.rate_limiter.tokens, 5.0)
        self.assertEqual(self.rate_limiter.resource_name, "test")
    
    def test_acquire_with_available_tokens(self) -> None:
        """Should successfully acquire token when tokens are available and decrement token count."""
        # Should succeed immediately
        self.assertTrue(self.rate_limiter.acquire())
        self.assertEqual(self.rate_limiter.tokens, 4.0)
    
    def test_acquire_with_token_refill(self) -> None:
        """Should refill tokens over elapsed time and allow new acquisition."""
        # Use all tokens
        for _ in range(5):
            self.assertTrue(self.rate_limiter.acquire())
        
        # No tokens left
        self.assertFalse(self.rate_limiter.acquire())
        
        # Advance time by 0.5 seconds (should get 5 new tokens)
        self.mock_time.return_value = 0.5
        self.assertTrue(self.rate_limiter.acquire())
        self.assertEqual(self.rate_limiter.tokens, 4.0)
    
    def test_acquire_with_timeout(self) -> None:
        """Should fail to acquire token when no tokens available and timeout is reached."""
        # Use all tokens
        for _ in range(5):
            self.assertTrue(self.rate_limiter.acquire())
        
        # Try to acquire with timeout
        self.assertFalse(self.rate_limiter.acquire(timeout=0.1))
    
    def test_reset(self) -> None:
        """Should restore rate limiter to initial state with full token capacity."""
        # Use some tokens
        self.rate_limiter.acquire()
        self.rate_limiter.acquire()
        
        # Reset
        self.rate_limiter.reset()
        
        # Should be back to initial state
        self.assertEqual(self.rate_limiter.tokens, self.rate_limiter.capacity)
    
    def test_burst_behavior(self) -> None:
        """Should allow burst of requests up to burst size and then enforce rate limit."""
        # Should be able to make 5 requests immediately (burst)
        for _ in range(5):
            self.assertTrue(self.rate_limiter.acquire())
        
        # No more tokens available
        self.assertFalse(self.rate_limiter.acquire())
        
        # Advance time by 0.1 seconds (should get 1 new token)
        self.mock_time.return_value = 0.1
        self.assertTrue(self.rate_limiter.acquire())
        self.assertFalse(self.rate_limiter.acquire())
    
    def test_environment_variable_configuration(self) -> None:
        """Should configure rate limiter using environment variables when provided."""
        with patch.dict('os.environ', {
            'RATE_LIMIT_TEST_REQUESTS_PER_SECOND': '20',
            'RATE_LIMIT_TEST_BURST_SIZE': '10'
        }):
            limiter = RateLimiter(resource_name="test")
            self.assertEqual(limiter.rate, 20.0)
            self.assertEqual(limiter.capacity, 10.0)
    
    def test_multiple_rate_limiters(self) -> None:
        """Should maintain independent token counts for different rate limiter instances."""
        limiter1 = RateLimiter(requests_per_second=10, burst_size=5, resource_name="test1")
        limiter2 = RateLimiter(requests_per_second=20, burst_size=10, resource_name="test2")
        
        # Use all tokens from limiter1
        for _ in range(5):
            self.assertTrue(limiter1.acquire())
        self.assertFalse(limiter1.acquire())
        
        # Limiter2 should still have all its tokens
        self.assertEqual(limiter2.tokens, 10.0)
        self.assertTrue(limiter2.acquire())

    def test_zero_rate_configuration(self) -> None:
        """Should raise ValueError when requests_per_second is set to zero."""
        with self.assertRaises(ValueError) as context:
            RateLimiter(requests_per_second=0, burst_size=5, resource_name="test")
        self.assertIn("requests_per_second", str(context.exception))
        self.assertIn("must be positive", str(context.exception))

    def test_zero_burst_size_configuration(self) -> None:
        """Should raise ValueError when burst_size is set to zero."""
        with self.assertRaises(ValueError) as context:
            RateLimiter(requests_per_second=10, burst_size=0, resource_name="test")
        self.assertIn("burst_size", str(context.exception))
        self.assertIn("must be positive", str(context.exception))

    @unittest.skipUnless(os.environ.get('RUN_PERFORMANCE_TESTS'), "Performance tests disabled")
    def test_acquire_performance(self) -> None:
        """Should maintain sub-millisecond performance for token acquisition under high load."""
        # Create a high-capacity rate limiter for testing
        limiter = RateLimiter(
            requests_per_second=1000.0,  # 1000 requests per second
            burst_size=1000,            # Allow burst of 1000 requests
            resource_name="perf_test"
        )
        
        # Warm up the rate limiter
        for _ in range(100):
            limiter.acquire()
        
        # Measure performance over 1000 acquisitions
        iterations = 1000
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            limiter.acquire()
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time_per_acquire = (total_time / iterations) * 1000  # Convert to milliseconds
        
        # Assert that average time per acquire is under 1ms
        self.assertLess(avg_time_per_acquire, 1.0, 
            f"Token acquisition took {avg_time_per_acquire:.3f}ms per operation, "
            f"exceeding 1ms threshold")

if __name__ == '__main__':
    unittest.main() 