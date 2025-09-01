"""
Performance optimization utilities for Gemini Prompt Optimizer
"""
import time
import asyncio
from typing import List, Dict, Any, Optional, Callable
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import json
import os
from models.exceptions import APIError

logger = logging.getLogger("gemini_optimizer.performance")

class ResponseCache:
    """Cache for API responses to avoid duplicate calls"""
    
    def __init__(self, cache_dir: str = "cache", max_size: int = 1000):
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        
        os.makedirs(cache_dir, exist_ok=True)
        self._load_cache()
    
    def _generate_key(self, prompt: str, model_name: str = "default") -> str:
        """Generate cache key from prompt and model"""
        content = f"{model_name}:{prompt}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _load_cache(self) -> None:
        """Load cache from disk"""
        try:
            cache_file = os.path.join(self.cache_dir, "response_cache.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cache = data.get('cache', {})
                    self.access_times = data.get('access_times', {})
                logger.info(f"Loaded {len(self.cache)} cached responses")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    
    def _save_cache(self) -> None:
        """Save cache to disk"""
        try:
            cache_file = os.path.join(self.cache_dir, "response_cache.json")
            data = {
                'cache': self.cache,
                'access_times': self.access_times
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get(self, prompt: str, model_name: str = "default") -> Optional[str]:
        """Get cached response"""
        key = self._generate_key(prompt, model_name)
        if key in self.cache:
            self.access_times[key] = time.time()
            logger.debug(f"Cache hit for key: {key[:8]}...")
            return self.cache[key]
        return None
    
    def put(self, prompt: str, response: str, model_name: str = "default") -> None:
        """Cache response"""
        key = self._generate_key(prompt, model_name)
        
        # Evict old entries if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = response
        self.access_times[key] = time.time()
        logger.debug(f"Cached response for key: {key[:8]}...")
        
        # Periodically save to disk
        if len(self.cache) % 10 == 0:
            self._save_cache()
    
    def _evict_oldest(self) -> None:
        """Evict oldest cache entry"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
        logger.debug(f"Evicted cache entry: {oldest_key[:8]}...")
    
    def clear(self) -> None:
        """Clear all cache"""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": getattr(self, '_hit_count', 0) / max(getattr(self, '_total_requests', 1), 1)
        }

class BatchProcessor:
    """Optimized batch processing for API calls"""
    
    def __init__(self, batch_size: int = 50, max_workers: int = 3, delay_between_batches: float = 1.0):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.delay_between_batches = delay_between_batches
        self.cache = ResponseCache()
        
    def process_batches(self, items: List[str], process_func: Callable, 
                       use_cache: bool = True) -> List[str]:
        """Process items in optimized batches"""
        if not items:
            return []
        
        logger.info(f"Processing {len(items)} items in batches of {self.batch_size}")
        
        # Split into batches
        batches = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]
        all_results = []
        
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i + 1}/{len(batches)} ({len(batch)} items)")
            
            try:
                # Check cache first if enabled
                if use_cache:
                    batch_results = self._process_batch_with_cache(batch, process_func)
                else:
                    batch_results = self._process_batch_direct(batch, process_func)
                
                all_results.extend(batch_results)
                
                # Delay between batches to avoid rate limits
                if i < len(batches) - 1:
                    time.sleep(self.delay_between_batches)
                    
            except Exception as e:
                logger.error(f"Failed to process batch {i + 1}: {e}")
                # Add empty results for failed batch
                all_results.extend([""] * len(batch))
        
        logger.info(f"Completed processing {len(all_results)} items")
        return all_results
    
    def _process_batch_with_cache(self, batch: List[str], process_func: Callable) -> List[str]:
        """Process batch with caching"""
        results = []
        uncached_items = []
        uncached_indices = []
        
        # Check cache for each item
        for i, item in enumerate(batch):
            cached_result = self.cache.get(item)
            if cached_result:
                results.append((i, cached_result))
            else:
                uncached_items.append(item)
                uncached_indices.append(i)
        
        # Process uncached items
        if uncached_items:
            logger.debug(f"Processing {len(uncached_items)} uncached items")
            uncached_results = self._process_batch_direct(uncached_items, process_func)
            
            # Cache results and add to results list
            for idx, result in zip(uncached_indices, uncached_results):
                self.cache.put(batch[idx], result)
                results.append((idx, result))
        
        # Sort by original index and return values
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]
    
    def _process_batch_direct(self, batch: List[str], process_func: Callable) -> List[str]:
        """Process batch directly without caching"""
        try:
            # Join batch items for single API call
            batch_input = "\n".join(batch)
            batch_result = process_func(batch_input)
            
            # Parse batch result
            if isinstance(batch_result, str):
                lines = batch_result.strip().split('\n')
                # Ensure we have the right number of results
                while len(lines) < len(batch):
                    lines.append("")
                return lines[:len(batch)]
            else:
                return [str(batch_result)] * len(batch)
                
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return [""] * len(batch)
    
    def process_parallel_batches(self, items: List[str], process_func: Callable,
                               use_cache: bool = True) -> List[str]:
        """Process batches in parallel (use with caution for API rate limits)"""
        if not items:
            return []
        
        logger.info(f"Processing {len(items)} items in parallel batches")
        
        # Split into batches
        batches = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]
        all_results = [None] * len(items)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit batch processing tasks
            future_to_batch = {}
            for i, batch in enumerate(batches):
                if use_cache:
                    future = executor.submit(self._process_batch_with_cache, batch, process_func)
                else:
                    future = executor.submit(self._process_batch_direct, batch, process_func)
                future_to_batch[future] = (i, batch)
            
            # Collect results
            for future in as_completed(future_to_batch):
                batch_idx, batch = future_to_batch[future]
                try:
                    batch_results = future.result()
                    start_idx = batch_idx * self.batch_size
                    for j, result in enumerate(batch_results):
                        if start_idx + j < len(all_results):
                            all_results[start_idx + j] = result
                except Exception as e:
                    logger.error(f"Parallel batch {batch_idx} failed: {e}")
                    # Fill with empty results
                    start_idx = batch_idx * self.batch_size
                    for j in range(len(batch)):
                        if start_idx + j < len(all_results):
                            all_results[start_idx + j] = ""
        
        # Filter out None values
        return [result or "" for result in all_results]

class MemoryOptimizer:
    """Memory optimization utilities"""
    
    @staticmethod
    def chunk_large_data(data: List[Any], chunk_size: int = 1000) -> List[List[Any]]:
        """Split large data into chunks for memory efficiency"""
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    @staticmethod
    def clear_large_variables(*variables) -> None:
        """Clear large variables to free memory"""
        for var in variables:
            if hasattr(var, 'clear'):
                var.clear()
            del var
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
                "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                "percent": process.memory_percent()
            }
        except ImportError:
            logger.warning("psutil not available, cannot get memory usage")
            return {}

class APIRateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int = 60, calls_per_hour: int = 1000):
        self.calls_per_minute = calls_per_minute
        self.calls_per_hour = calls_per_hour
        self.minute_calls = []
        self.hour_calls = []
    
    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded"""
        current_time = time.time()
        
        # Clean old calls
        self.minute_calls = [t for t in self.minute_calls if current_time - t < 60]
        self.hour_calls = [t for t in self.hour_calls if current_time - t < 3600]
        
        # Check minute limit
        if len(self.minute_calls) >= self.calls_per_minute:
            wait_time = 60 - (current_time - self.minute_calls[0])
            if wait_time > 0:
                logger.info(f"Rate limit: waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
        
        # Check hour limit
        if len(self.hour_calls) >= self.calls_per_hour:
            wait_time = 3600 - (current_time - self.hour_calls[0])
            if wait_time > 0:
                logger.info(f"Hourly rate limit: waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
        
        # Record this call
        current_time = time.time()
        self.minute_calls.append(current_time)
        self.hour_calls.append(current_time)

class PerformanceMonitor:
    """Monitor performance metrics during optimization"""
    
    def __init__(self):
        self.metrics = {
            "api_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_processing_time": 0,
            "average_response_time": 0,
            "batch_processing_times": [],
            "memory_usage": []
        }
        self.start_time = None
    
    def start_monitoring(self) -> None:
        """Start performance monitoring"""
        self.start_time = time.time()
        self.metrics = {key: 0 if isinstance(value, (int, float)) else [] 
                       for key, value in self.metrics.items()}
    
    def record_api_call(self, response_time: float) -> None:
        """Record API call metrics"""
        self.metrics["api_calls"] += 1
        self.metrics["total_processing_time"] += response_time
        self.metrics["average_response_time"] = (
            self.metrics["total_processing_time"] / self.metrics["api_calls"]
        )
    
    def record_cache_hit(self) -> None:
        """Record cache hit"""
        self.metrics["cache_hits"] += 1
    
    def record_cache_miss(self) -> None:
        """Record cache miss"""
        self.metrics["cache_misses"] += 1
    
    def record_batch_time(self, batch_time: float) -> None:
        """Record batch processing time"""
        self.metrics["batch_processing_times"].append(batch_time)
    
    def record_memory_usage(self) -> None:
        """Record current memory usage"""
        memory_info = MemoryOptimizer.get_memory_usage()
        if memory_info:
            self.metrics["memory_usage"].append(memory_info)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        cache_total = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        cache_hit_rate = self.metrics["cache_hits"] / max(cache_total, 1)
        
        batch_times = self.metrics["batch_processing_times"]
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        
        return {
            "execution_time": total_time,
            "api_calls": self.metrics["api_calls"],
            "average_response_time": self.metrics["average_response_time"],
            "cache_hit_rate": cache_hit_rate,
            "total_cache_requests": cache_total,
            "average_batch_time": avg_batch_time,
            "total_batches": len(batch_times),
            "api_calls_per_second": self.metrics["api_calls"] / max(total_time, 1),
            "memory_peak": max([m.get("rss_mb", 0) for m in self.metrics["memory_usage"]], default=0)
        }
    
    def print_performance_summary(self) -> None:
        """Print performance summary"""
        report = self.get_performance_report()
        
        print("\n" + "="*50)
        print("📊 PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Total Execution Time: {report['execution_time']:.2f} seconds")
        print(f"API Calls Made: {report['api_calls']}")
        print(f"Average Response Time: {report['average_response_time']:.2f} seconds")
        print(f"Cache Hit Rate: {report['cache_hit_rate']:.1%}")
        print(f"Average Batch Time: {report['average_batch_time']:.2f} seconds")
        print(f"API Calls per Second: {report['api_calls_per_second']:.2f}")
        
        if report['memory_peak'] > 0:
            print(f"Peak Memory Usage: {report['memory_peak']:.1f} MB")
        
        print("="*50)

# Decorator for performance monitoring
def monitor_performance(monitor: PerformanceMonitor):
    """Decorator to monitor function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                response_time = time.time() - start_time
                monitor.record_api_call(response_time)
                return result
            except Exception as e:
                response_time = time.time() - start_time
                monitor.record_api_call(response_time)
                raise
        return wrapper
    return decorator