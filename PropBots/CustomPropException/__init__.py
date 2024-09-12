import sys
import logging
from PropBot.logger import logging



def format_error_message(error, error_detail: sys, additional_info=None):
    """Formats detailed error message with file name, line number, and error details."""
    exc_type, exc_value, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    function_name = exc_tb.tb_frame.f_code.co_name

    # Additional context if provided
    context = f"Context: {additional_info}" if additional_info else ""

    error_message = (
        f"Error occurred in function: {function_name}\n"
        f"File: {file_name}\n"
        f"Line: {line_number}\n"
        f"Error: {str(error)}\n"
        f"{context}"
    )
    return error_message

class PropBotException(Exception):
    """Custom exception class for handling errors in compliance detection systems."""

    def __init__(self, error_message, error_detail: sys, additional_info=None, log_error=True):
        """
        Initialize ComplianceDetectionException with a formatted error message.

        :param error_message: The original error message.
        :param error_detail: The sys module for detailed error information.
        :param additional_info: Optional additional information for context (e.g., model, input data).
        :param log_error: Whether to log the error automatically.
        """
        formatted_message = format_error_message(error_message, error_detail, additional_info)
        super().__init__(formatted_message)
        self.error_message = formatted_message

        # Log the error if specified
        if log_error:
            logging.error(self.error_message)

    def __str__(self):
        return self.error_message

    def log_exception(self, custom_message=None):
        """Log the exception with an optional custom message."""
        if custom_message:
            logging.error(f"{custom_message} | {self.error_message}")
        else:
            logging.error(self.error_message)

# Usage Example
if __name__ == '__main__':
    try:
        # Example code that might raise an exception
        raise ValueError("Sample error for demonstration")
    except Exception as e:
        raise PropBotException("An error occurred in the AI process", sys, additional_info={"model": "GPT-4", "input": "sample text"}).log_exception()
