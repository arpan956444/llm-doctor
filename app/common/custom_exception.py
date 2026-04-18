import sys

class CustomException(Exception):
    def __init__(self, message: str, error_details: Exception = None):
        # FIX: Change 'get_detail_error_message' to 'get_detailed_error_message'
        self.error_message = self.get_detailed_error_message(message, error_details)
        super().__init__(self.error_message)

    def get_detailed_error_message(self, message, error_detail):
        _, _, exc_tb = sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown File"
        line_number = exc_tb.tb_lineno if exc_tb else "Unknown Line"

        return (
            f"{message} | Error: {error_detail} | "
            f"File: {file_name} | Line: {line_number}"
        )

    def __str__(self):
        return self.error_message