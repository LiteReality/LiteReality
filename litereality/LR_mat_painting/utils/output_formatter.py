"""
Output formatting utilities for cleaner terminal output in material painting pipeline.
Provides consistent formatting similar to preprocessing pipeline.
"""

import sys
import time
from datetime import datetime
from contextlib import contextmanager

class OutputFormatter:
    """Utility class for formatting output messages consistently"""
    
    @staticmethod
    def print_header(title, emoji="🎨"):
        """Print a boxed header"""
        width = 63
        print(f"\n╔{'═' * width}╗")
        print(f"║ {emoji} {title:<{width-len(emoji)-3}} ║")
        print(f"╚{'═' * width}╝")
    
    @staticmethod
    def print_step_header(step_num, total_steps, title, emoji="🖼️", time_estimate=""):
        """Print a step header"""
        time_str = f" (~ {time_estimate})" if time_estimate else ""
        print(f"\n[Step {step_num}/{total_steps}] {emoji} {title}{time_str}")
        print("-" * 57)
    
    @staticmethod
    def print_success(message, indent=0):
        """Print a success message with checkmark"""
        indent_str = "  " * indent
        print(f"{indent_str}✓ {message}")
    
    @staticmethod
    def print_info(message, indent=0):
        """Print an info message"""
        indent_str = "  " * indent
        print(f"{indent_str}ℹ️  {message}")
    
    @staticmethod
    def print_warning(message, indent=0):
        """Print a warning message"""
        indent_str = "  " * indent
        print(f"{indent_str}⚠️  {message}")
    
    @staticmethod
    def print_error(message, indent=0):
        """Print an error message"""
        indent_str = "  " * indent
        print(f"{indent_str}❌ {message}")
    
    @staticmethod
    def print_summary(title, items, total_time=None):
        """Print a summary box"""
        width = 63
        print(f"\n╔{'═' * width}╗")
        print(f"║ ✓ {title:<{width-4}} ║")
        print(f"╚{'═' * width}╝")
        for key, value in items.items():
            print(f"  - {key}: {value}")
        if total_time:
            print(f"Total time: {total_time:.2f}s")
    
    @staticmethod
    def suppress_output():
        """Context manager to suppress verbose output"""
        class SuppressOutput:
            def __enter__(self):
                self.original_stdout = sys.stdout
                self.original_stderr = sys.stderr
                sys.stdout = open('/dev/null', 'w')
                sys.stderr = open('/dev/null', 'w')
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = self.original_stdout
                sys.stderr = self.original_stderr
                return False
        
        return SuppressOutput()
    
    @staticmethod
    @contextmanager
    def timed_operation(description, show_time=True):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            if show_time:
                print(f"  ✓ {description} ({elapsed:.1f}s)")

# Global instance for easy access
formatter = OutputFormatter()



