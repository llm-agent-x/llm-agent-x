import random
import inspect
import datetime
import uuid
import json
import csv
import io

# --- Global Configuration & Helper Functions ---
_SUCCESS_RATE = 0.8  # 80% success rate for operations


def _random_success_or_failure(success_rate=_SUCCESS_RATE):
    """Determines if an operation succeeds or fails based on a success rate."""
    return random.random() < success_rate


def _generate_random_string(length=10, charset="abcdefghijklmnopqrstuvwxyz0123456789"):
    """Generates a random string of a given length."""
    return "".join(random.choice(charset) for _ in range(length))


def _generate_random_token(prefix="tok_"):
    """Generates a random token string."""
    return f"{prefix}{_generate_random_string(32)}"


def _generate_random_id(prefix="id_"):
    """Generates a random ID string."""
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def _get_random_error_message(action_verb):
    """Returns a generic random error message for a failed action."""
    errors = [
        f"Unable to {action_verb} due to a network timeout.",
        f"An unexpected internal error occurred while trying to {action_verb}.",
        f"Access denied: insufficient permissions to {action_verb}.",
        f"The resource required to {action_verb} was not found.",
        f"The {action_verb} operation was interrupted by the system.",
        f"Failed to {action_verb}: invalid parameters provided.",
    ]
    return random.choice(errors)


# --- Decorator for adding tags to functions ---
def add_tags(*tags):
    """
    Decorator to add a 'tags' attribute to a function.
    The primary category (class name) will be added automatically.
    """

    def decorator(func):
        func.tags = list(tags)
        return func

    return decorator


# --- Base Class for Tasks ---
class BaseAgentTask:
    """
    Base class for agent tasks, providing common utilities.
    Subclasses will represent categories of tasks.
    """

    # Helper methods could be added here if they need 'cls' or 'self'
    # For now, module-level helpers are used.
    pass


# --- Task Category Classes ---


class CommunicationTasks(BaseAgentTask):
    """Tasks related to sending and receiving communications."""

    @staticmethod
    @add_tags("COMM", "INET")
    def send_email(sender_email, recipient_emails, subject, body):
        """
        Simulates sending an email.

        Args:
            sender_email (str): The email address of the sender.
            recipient_emails (list[str]): A list of recipient email addresses.
            subject (str): The subject line of the email.
            body (str): The main content of the email.

        Returns:
            dict: A dictionary containing the status of the operation and details.
                On success: {"status": "success", "message": "Email sent successfully",
                             "message_id": str, "sender": str, "recipients": list, "subject": str}
                On failure: {"status": "failure", "error": str, "details": str}
        """
        action_verb = "send email"
        print(
            f"Attempting to {action_verb} from {sender_email} to {recipient_emails} with subject '{subject}'..."
        )
        if _random_success_or_failure():
            result = {
                "status": "success",
                "message": "Email sent successfully",
                "message_id": _generate_random_id("email_"),
                "sender": sender_email,
                "recipients": recipient_emails,
                "subject": subject,
            }
        else:
            result = {
                "status": "failure",
                "error": "Unable to connect to SMTP server",
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("COMM")
    def send_sms(phone_number, message_text):
        """
        Simulates sending an SMS message.

        Args:
            phone_number (str): The recipient's phone number.
            message_text (str): The content of the SMS.

        Returns:
            dict: A dictionary containing the status of the operation.
                On success: {"status": "success", "message": "SMS sent successfully",
                             "message_id": str, "phone_number": str, "delivery_status": str}
                On failure: {"status": "failure", "error": str, "phone_number": str}
        """
        action_verb = "send SMS"
        print(f"Attempting to {action_verb} to {phone_number}...")
        if _random_success_or_failure():
            result = {
                "status": "success",
                "message": "SMS sent successfully",
                "message_id": _generate_random_id("sms_"),
                "phone_number": phone_number,
                "delivery_status": random.choice(["DELIVERED", "PENDING"]),
            }
        else:
            result = {
                "status": "failure",
                "error": "SMS gateway unavailable or number invalid",
                "phone_number": phone_number,
            }
        print(result)
        return result

    @staticmethod
    @add_tags("COMM", "INET")
    def send_push_notification(device_id, title, message_body):
        """
        Simulates sending a push notification to a device.

        Args:
            device_id (str): The unique identifier of the target device.
            title (str): The title of the push notification.
            message_body (str): The main content of the notification.

        Returns:
            dict: A dictionary containing the status of the operation.
                On success: {"status": "success", "notification_id": str, "device_id": str, "title": str}
                On failure: {"status": "failure", "error": str, "device_id": str}
        """
        action_verb = "send push notification"
        print(
            f"Attempting to {action_verb} to device {device_id} with title '{title}'..."
        )
        if _random_success_or_failure():
            result = {
                "status": "success",
                "message": "Push notification sent successfully",
                "notification_id": _generate_random_id("push_"),
                "device_id": device_id,
                "title": title,
            }
        else:
            result = {
                "status": "failure",
                "error": "Push notification delivery failed (e.g., invalid token, service down)",
                "device_id": device_id,
            }
        print(result)
        return result

    @staticmethod
    @add_tags("COMM", "CALENDAR")
    def schedule_meeting_invite(
        organizer_email,
        attendee_emails,
        subject,
        start_time_iso,
        end_time_iso,
        location="Virtual",
    ):
        """
        Simulates scheduling a meeting and sending invites.

        Args:
            organizer_email (str): Email of the meeting organizer.
            attendee_emails (list[str]): List of attendee email addresses.
            subject (str): Meeting subject.
            start_time_iso (str): Meeting start time in ISO 8601 format.
            end_time_iso (str): Meeting end time in ISO 8601 format.
            location (str, optional): Meeting location. Defaults to "Virtual".

        Returns:
            dict: Status of the scheduling operation.
                On success: {"status": "success", "message": "Meeting scheduled and invites sent",
                             "meeting_id": str, "organizer": str, "attendees": list}
                On failure: {"status": "failure", "error": str}
        """
        action_verb = "schedule meeting invite"
        print(
            f"Attempting to {action_verb} for '{subject}' organized by {organizer_email}..."
        )
        if _random_success_or_failure():
            result = {
                "status": "success",
                "message": "Meeting scheduled and invites sent successfully",
                "meeting_id": _generate_random_id("meet_"),
                "organizer": organizer_email,
                "attendees": attendee_emails,
                "subject": subject,
                "start_time": start_time_iso,
                "end_time": end_time_iso,
                "location": location,
            }
        else:
            result = {
                "status": "failure",
                "error": "Failed to schedule meeting (e.g., calendar service unavailable, conflicting times)",
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("COMM", "SOCIAL", "INET")
    def post_social_media_update(platform, account_id, message_content, image_url=None):
        """
        Simulates posting an update to a social media platform.

        Args:
            platform (str): Name of the social media platform (e.g., "Twitter", "Facebook").
            account_id (str): The user's account ID or handle on the platform.
            message_content (str): The text content of the post.
            image_url (str, optional): URL of an image to include in the post.

        Returns:
            dict: Status of the social media post.
                On success: {"status": "success", "message": "Posted successfully", "post_id": str, "platform": str}
                On failure: {"status": "failure", "error": str, "platform": str}
        """
        action_verb = f"post to {platform}"
        print(f"Attempting to {action_verb} for account {account_id}...")
        if _random_success_or_failure():
            result = {
                "status": "success",
                "message": f"Update posted successfully to {platform}",
                "post_id": _generate_random_id(f"{platform.lower()}_post_"),
                "platform": platform,
                "account_id": account_id,
                "content_preview": (
                    message_content[:50] + "..."
                    if len(message_content) > 50
                    else message_content
                ),
            }
        else:
            result = {
                "status": "failure",
                "error": f"Failed to post to {platform} (e.g., API limit, invalid credentials, content policy violation)",
                "platform": platform,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("COMM", "CHAT", "INET")
    def send_chat_message(service_name, channel_id, user_token, message_text):
        """
        Simulates sending a message to a chat service like Slack or Teams.

        Args:
            service_name (str): Name of the chat service (e.g., "Slack", "Microsoft Teams").
            channel_id (str): Identifier for the channel or direct message.
            user_token (str): Authentication token for the user.
            message_text (str): The message content.

        Returns:
            dict: Status of the message sending operation.
                On success: {"status": "success", "message_id": str, "service": str, "channel": str}
                On failure: {"status": "failure", "error": str, "service": str, "channel": str}
        """
        action_verb = f"send chat message to {service_name}"
        print(f"Attempting to {action_verb} in channel {channel_id}...")
        if _random_success_or_failure():
            result = {
                "status": "success",
                "message": "Chat message sent successfully.",
                "message_id": _generate_random_id(f"{service_name.lower()}_msg_"),
                "service": service_name,
                "channel": channel_id,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
        else:
            result = {
                "status": "failure",
                "error": f"Failed to send message on {service_name} (e.g., invalid token, channel not found, rate limit).",
                "service": service_name,
                "channel": channel_id,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result


class FileManagementTasks(BaseAgentTask):
    """Tasks related to file system operations."""

    @staticmethod
    @add_tags("FM")
    def read_file_content(file_path):
        """
        Simulates reading the content of a file.

        Args:
            file_path (str): The path to the file to be read.

        Returns:
            dict: A dictionary containing the status and file content.
                On success: {"status": "success", "file_path": str, "content": str, "size_bytes": int}
                On failure: {"status": "failure", "error": str, "file_path": str}
        """
        action_verb = "read file content"
        print(f"Attempting to {action_verb} from {file_path}...")
        if _random_success_or_failure():
            simulated_size = random.randint(100, 10240)
            result = {
                "status": "success",
                "file_path": file_path,
                "content": f"Simulated content of {file_path}. File size: {simulated_size} bytes. Lorem ipsum dolor sit amet...",
                "size_bytes": simulated_size,
                "encoding": "utf-8",
            }
        else:
            result = {
                "status": "failure",
                "error": "File not found or access denied",
                "file_path": file_path,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("FM")
    def write_to_file(file_path, content, mode="w"):
        """
        Simulates writing content to a file.

        Args:
            file_path (str): The path to the file to be written.
            content (str): The content to write to the file.
            mode (str, optional): File opening mode ('w' for write, 'a' for append). Defaults to 'w'.

        Returns:
            dict: A dictionary containing the status of the write operation.
                On success: {"status": "success", "message": "File written successfully", "file_path": str, "bytes_written": int}
                On failure: {"status": "failure", "error": str, "file_path": str}
        """
        action_verb = "write to file"
        print(f"Attempting to {action_verb} at {file_path} (mode: {mode})...")
        if _random_success_or_failure():
            bytes_written = (
                len(content.encode("utf-8"))
                if isinstance(content, str)
                else len(content)
            )
            result = {
                "status": "success",
                "message": f"File {'written' if mode == 'w' else 'appended'} successfully.",
                "file_path": file_path,
                "bytes_written": bytes_written,
                "mode": mode,
            }
        else:
            result = {
                "status": "failure",
                "error": "Failed to write to file (e.g., disk full, permissions error)",
                "file_path": file_path,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("FM")
    def delete_file(file_path):
        """
        Simulates deleting a file.

        Args:
            file_path (str): The path to the file to be deleted.

        Returns:
            dict: Status of the deletion.
                On success: {"status": "success", "message": "File deleted successfully", "file_path": str}
                On failure: {"status": "failure", "error": str, "file_path": str}
        """
        action_verb = "delete file"
        print(f"Attempting to {action_verb} {file_path}...")
        if _random_success_or_failure():
            result = {
                "status": "success",
                "message": "File deleted successfully.",
                "file_path": file_path,
            }
        else:
            result = {
                "status": "failure",
                "error": "Failed to delete file (e.g., file not found, in use, permissions error)",
                "file_path": file_path,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("FM")
    def list_directory_contents(directory_path):
        """
        Simulates listing the contents of a directory.

        Args:
            directory_path (str): The path to the directory.

        Returns:
            dict: Status and directory contents.
                On success: {"status": "success", "directory_path": str, "contents": list[dict]}
                            (each dict: {"name": str, "type": "file"|"directory", "size": int (for files)})
                On failure: {"status": "failure", "error": str, "directory_path": str}
        """
        action_verb = "list directory contents"
        print(f"Attempting to {action_verb} for {directory_path}...")
        if _random_success_or_failure():
            num_items = random.randint(0, 10)
            contents = []
            for i in range(num_items):
                item_type = random.choice(["file", "directory"])
                item_name = f"{random.choice(['document', 'image', 'folder', 'archive', 'script'])}_{_generate_random_string(5)}"
                if item_type == "file":
                    item_name += random.choice([".txt", ".jpg", ".dat", ".zip", ".py"])
                    contents.append(
                        {
                            "name": item_name,
                            "type": "file",
                            "size_bytes": random.randint(100, 5000000),
                        }
                    )
                else:
                    contents.append({"name": item_name, "type": "directory"})
            result = {
                "status": "success",
                "directory_path": directory_path,
                "contents": contents,
                "count": len(contents),
            }
        else:
            result = {
                "status": "failure",
                "error": "Failed to list directory (e.g., path does not exist, not a directory, permissions error)",
                "directory_path": directory_path,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("FM")
    def create_directory(directory_path, create_parents=False):
        """
        Simulates creating a new directory.

        Args:
            directory_path (str): The path where the new directory should be created.
            create_parents (bool): If True, create parent directories if they don't exist.

        Returns:
            dict: Status of the directory creation.
                On success: {"status": "success", "message": "Directory created", "directory_path": str}
                On failure: {"status": "failure", "error": str, "directory_path": str}
        """
        action_verb = "create directory"
        print(
            f"Attempting to {action_verb} at {directory_path} (create_parents: {create_parents})..."
        )
        if _random_success_or_failure():
            result = {
                "status": "success",
                "message": "Directory created successfully.",
                "directory_path": directory_path,
            }
        else:
            result = {
                "status": "failure",
                "error": "Failed to create directory (e.g., path already exists, permissions error, invalid path)",
                "directory_path": directory_path,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("FM", "UTIL")
    def compress_files(source_paths, output_archive_path, compression_format="zip"):
        """
        Simulates compressing one or more files/directories into an archive.

        Args:
            source_paths (list[str]): List of paths to files or directories to compress.
            output_archive_path (str): Path for the resulting archive file.
            compression_format (str, optional): Format like "zip", "tar.gz". Defaults to "zip".

        Returns:
            dict: Status of the compression.
                On success: {"status": "success", "archive_path": str, "format": str, "original_size_bytes": int, "compressed_size_bytes": int}
                On failure: {"status": "failure", "error": str}
        """
        action_verb = "compress files"
        print(
            f"Attempting to {action_verb} {source_paths} into {output_archive_path} (format: {compression_format})..."
        )
        if _random_success_or_failure():
            original_size = random.randint(10000, 1000000)
            compressed_size = int(original_size * random.uniform(0.3, 0.8))
            result = {
                "status": "success",
                "message": "Files compressed successfully.",
                "archive_path": output_archive_path,
                "format": compression_format,
                "source_file_count": len(source_paths),
                "original_total_size_bytes": original_size,
                "compressed_size_bytes": compressed_size,
            }
        else:
            result = {
                "status": "failure",
                "error": f"Compression failed (e.g., source files not found, disk space issue, invalid format '{compression_format}')",
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("FM", "UTIL")
    def decompress_archive(archive_path, destination_path, archive_format="zip"):
        """
        Simulates decompressing an archive file.

        Args:
            archive_path (str): Path to the archive file (e.g., .zip, .tar.gz).
            destination_path (str): Path to the directory where contents should be extracted.
            archive_format (str, optional): The format of the archive. Auto-detected if possible.

        Returns:
            dict: Status of the decompression.
                On success: {"status": "success", "destination_path": str, "files_extracted_count": int}
                On failure: {"status": "failure", "error": str, "archive_path": str}
        """
        action_verb = "decompress archive"
        print(f"Attempting to {action_verb} {archive_path} to {destination_path}...")
        if _random_success_or_failure():
            files_extracted_count = random.randint(1, 20)
            result = {
                "status": "success",
                "message": "Archive decompressed successfully.",
                "archive_path": archive_path,
                "destination_path": destination_path,
                "files_extracted_count": files_extracted_count,
                "format_detected": archive_format,
            }
        else:
            result = {
                "status": "failure",
                "error": "Decompression failed (e.g., corrupted archive, insufficient disk space, wrong format)",
                "archive_path": archive_path,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("FM")
    def get_file_metadata(file_path):
        """
        Simulates retrieving metadata for a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            dict: Status and file metadata.
                On success: {"status": "success", "file_path": str, "metadata": {"size_bytes": int,
                             "type": str, "created_at_iso": str, "modified_at_iso": str, "permissions": str}}
                On failure: {"status": "failure", "error": str, "file_path": str}
        """
        action_verb = "get file metadata"
        print(f"Attempting to {action_verb} for {file_path}...")
        if _random_success_or_failure():
            now = datetime.datetime.utcnow()
            created_at = (
                now - datetime.timedelta(days=random.randint(1, 365))
            ).isoformat() + "Z"
            modified_at = (
                now - datetime.timedelta(days=random.randint(0, 30))
            ).isoformat() + "Z"
            result = {
                "status": "success",
                "file_path": file_path,
                "metadata": {
                    "size_bytes": random.randint(
                        10, 1024 * 1024 * 10
                    ),  # 10 bytes to 10MB
                    "mime_type": random.choice(
                        [
                            "text/plain",
                            "image/jpeg",
                            "application/pdf",
                            "application/zip",
                        ]
                    ),
                    "created_at_iso": created_at,
                    "modified_at_iso": modified_at,
                    "owner": _generate_random_string(8, "abcdefghijklmnopqrstuvwxyz"),
                    "permissions": random.choice(
                        ["rw-r--r--", "rwxr-xr-x", "rw-rw----"]
                    ),
                },
            }
        else:
            result = {
                "status": "failure",
                "error": "Failed to retrieve metadata (e.g., file not found, access denied)",
                "file_path": file_path,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result


class DataProcessingTasks(BaseAgentTask):
    """Tasks related to parsing, transforming, and analyzing data."""

    @staticmethod
    @add_tags("DP", "PARSE")
    def parse_csv_data(csv_string_content, delimiter=",", has_header=True):
        """
        Simulates parsing CSV data from a string.

        Args:
            csv_string_content (str): The CSV data as a string.
            delimiter (str, optional): The delimiter used in the CSV. Defaults to ','.
            has_header (bool, optional): Whether the CSV has a header row. Defaults to True.

        Returns:
            dict: Status and parsed data.
                On success: {"status": "success", "records": list[dict] or list[list], "num_records": int, "num_columns": int, "headers": list (if has_header)}
                On failure: {"status": "failure", "error": str, "input_preview": str}
        """
        action_verb = "parse CSV data"
        print(f"Attempting to {action_verb}...")
        if _random_success_or_failure():
            try:
                # Simulate parsing
                f = io.StringIO(csv_string_content)
                reader = csv.reader(f, delimiter=delimiter)
                headers = []
                records_list = []

                if has_header:
                    headers = next(reader, [])
                    for row in reader:
                        records_list.append(dict(zip(headers, row)))
                else:
                    for row in reader:
                        records_list.append(list(row))

                num_columns = (
                    len(headers)
                    if has_header and headers
                    else (len(records_list[0]) if records_list else 0)
                )

                result = {
                    "status": "success",
                    "message": "CSV data parsed successfully.",
                    "records": records_list,
                    "num_records": len(records_list),
                    "num_columns": num_columns,
                }
                if has_header:
                    result["headers"] = headers
            except Exception as e:  # Catch potential errors from simple parsing attempt
                result = {
                    "status": "failure",
                    "error": f"CSV parsing error during simulation: {str(e)}",
                    "input_preview": csv_string_content[:100] + "...",
                }
        else:
            result = {
                "status": "failure",
                "error": "Failed to parse CSV (e.g., malformed CSV, incorrect delimiter)",
                "input_preview": csv_string_content[:100] + "...",
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("DP", "PARSE")
    def parse_json_data(json_string_content):
        """
        Simulates parsing JSON data from a string.

        Args:
            json_string_content (str): The JSON data as a string.

        Returns:
            dict: Status and parsed data.
                On success: {"status": "success", "data": dict or list, "structure_type": str}
                On failure: {"status": "failure", "error": str, "input_preview": str}
        """
        action_verb = "parse JSON data"
        print(f"Attempting to {action_verb}...")
        if _random_success_or_failure():
            try:
                parsed_data = json.loads(json_string_content)
                result = {
                    "status": "success",
                    "message": "JSON data parsed successfully.",
                    "data": parsed_data,
                    "structure_type": (
                        "object"
                        if isinstance(parsed_data, dict)
                        else (
                            "array"
                            if isinstance(parsed_data, list)
                            else str(type(parsed_data))
                        )
                    ),
                }
            except json.JSONDecodeError as e:
                result = {  # This case means the input JSON was bad, even if _random_success_or_failure was true
                    "status": "failure",
                    "error": f"Malformed JSON: {str(e)}",
                    "input_preview": json_string_content[:100] + "...",
                }
        else:
            result = {
                "status": "failure",
                "error": "Failed to parse JSON (e.g., internal parser error, resource limit)",
                "input_preview": json_string_content[:100] + "...",
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("DP", "VALIDATE")
    def validate_data_against_schema(data_object, schema_definition):
        """
        Simulates validating a data object against a schema.

        Args:
            data_object (dict or list): The data to validate.
            schema_definition (dict): The schema definition (e.g., JSON Schema).

        Returns:
            dict: Status of validation.
                On success: {"status": "success", "message": "Data is valid", "is_valid": True}
                On success (invalid): {"status": "success", "message": "Data is invalid", "is_valid": False, "validation_errors": list}
                On failure: {"status": "failure", "error": str}
        """
        action_verb = "validate data against schema"
        print(f"Attempting to {action_verb}...")
        if (
            _random_success_or_failure()
        ):  # Simulates the validation process itself succeeding
            # Now, simulate whether the data actually IS valid or not
            is_actually_valid = random.random() > 0.3  # 70% chance data is valid
            if is_actually_valid:
                result = {
                    "status": "success",
                    "message": "Data is valid against the provided schema.",
                    "is_valid": True,
                }
            else:
                validation_errors = [
                    {"path": "user.email", "message": "must be a valid email address"},
                    {"path": "user.age", "message": "must be a positive integer"},
                ]
                result = {
                    "status": "success",  # The validation *operation* succeeded, but data was invalid
                    "message": "Data is invalid according to the schema.",
                    "is_valid": False,
                    "validation_errors": random.sample(
                        validation_errors, k=random.randint(1, len(validation_errors))
                    ),
                }
        else:
            result = {
                "status": "failure",
                "error": "Schema validation process failed (e.g., invalid schema, validator unavailable)",
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("DP", "TRANSFORM")
    def sort_data_records(records, sort_key, reverse_order=False):
        """
        Simulates sorting a list of records (dictionaries or objects).

        Args:
            records (list): A list of records (e.g., list of dicts).
            sort_key (str or callable): The key or function to sort by.
            reverse_order (bool, optional): Sort in descending order if True. Defaults to False.

        Returns:
            dict: Status and sorted data.
                On success: {"status": "success", "sorted_records": list, "sort_key_used": str, "order": str}
                On failure: {"status": "failure", "error": str}
        """
        action_verb = "sort data records"
        print(
            f"Attempting to {action_verb} by '{sort_key}' (reverse: {reverse_order})..."
        )
        if _random_success_or_failure():
            # Basic simulation of sorting, assuming records are dicts and sort_key is a string
            try:
                if isinstance(records, list) and all(
                    isinstance(r, dict) for r in records
                ):
                    sorted_records = sorted(
                        records, key=lambda x: x.get(sort_key), reverse=reverse_order
                    )
                else:  # Fallback for other types or if key is complex
                    sorted_records = sorted(
                        records, reverse=reverse_order
                    )  # Simplistic sort
                result = {
                    "status": "success",
                    "message": "Data sorted successfully.",
                    "sorted_records": sorted_records,
                    "original_count": len(records),
                    "sort_key_used": str(sort_key),
                    "order": "descending" if reverse_order else "ascending",
                }
            except TypeError:  # If sort_key is not applicable to items
                result = {
                    "status": "failure",
                    "error": f"Sorting failed: Incompatible data types for sort key '{sort_key}'.",
                }
        else:
            result = {
                "status": "failure",
                "error": "Data sorting process failed (e.g., memory limit, unstable sort algorithm issue)",
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("DP", "FILTER")
    def filter_data_records(records, filter_criteria_description):
        """
        Simulates filtering a list of records based on some criteria.
        The criteria is a string description for simulation purposes.

        Args:
            records (list): A list of records (e.g., list of dicts).
            filter_criteria_description (str): A human-readable description of the filter
                                               (e.g., "age > 30 and city == 'New York'").

        Returns:
            dict: Status and filtered data.
                On success: {"status": "success", "filtered_records": list, "original_count": int, "filtered_count": int}
                On failure: {"status": "failure", "error": str}
        """
        action_verb = "filter data records"
        print(
            f"Attempting to {action_verb} with criteria: '{filter_criteria_description}'..."
        )
        if _random_success_or_failure():
            original_count = len(records)
            # Simulate filtering by randomly selecting a subset
            if original_count > 0:
                num_to_keep = random.randint(0, original_count)
                filtered_records = (
                    random.sample(records, num_to_keep)
                    if num_to_keep < original_count
                    else list(records)
                )
            else:
                filtered_records = []

            result = {
                "status": "success",
                "message": "Data filtered successfully.",
                "filtered_records": filtered_records,
                "original_count": original_count,
                "filtered_count": len(filtered_records),
                "criteria_applied": filter_criteria_description,
            }
        else:
            result = {
                "status": "failure",
                "error": "Data filtering process failed (e.g., invalid filter syntax, resource constraints)",
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("DP", "AGGREGATE")
    def calculate_data_statistics(
        numeric_data_list,
        operations=["mean", "median", "stddev", "min", "max", "sum", "count"],
    ):
        """
        Simulates calculating descriptive statistics for a list of numbers.

        Args:
            numeric_data_list (list[float or int]): A list of numerical data.
            operations (list[str]): List of statistics to compute.

        Returns:
            dict: Status and computed statistics.
                On success: {"status": "success", "statistics": dict, "count": int}
                On failure: {"status": "failure", "error": str}
        """
        action_verb = "calculate data statistics"
        print(
            f"Attempting to {action_verb} for {len(numeric_data_list)} data points..."
        )
        if (
            not all(isinstance(x, (int, float)) for x in numeric_data_list)
            and numeric_data_list
        ):
            result = {
                "status": "failure",
                "error": "Invalid input: data list must contain only numbers.",
            }
        elif _random_success_or_failure():
            stats = {}
            count = len(numeric_data_list)
            if count > 0:
                if "count" in operations:
                    stats["count"] = count
                if "sum" in operations:
                    stats["sum"] = sum(numeric_data_list)
                if "mean" in operations and count > 0:
                    stats["mean"] = sum(numeric_data_list) / count
                if "min" in operations:
                    stats["min"] = min(numeric_data_list)
                if "max" in operations:
                    stats["max"] = max(numeric_data_list)
                # Simplified median and stddev for simulation
                if "median" in operations:
                    sorted_list = sorted(numeric_data_list)
                    if count % 2 == 1:
                        stats["median"] = sorted_list[count // 2]
                    else:
                        stats["median"] = (
                            sorted_list[count // 2 - 1] + sorted_list[count // 2]
                        ) / 2
                if "stddev" in operations and count > 1:  # Basic stddev simulation
                    mean_val = stats.get("mean", sum(numeric_data_list) / count)
                    variance = sum([(x - mean_val) ** 2 for x in numeric_data_list]) / (
                        count - 1
                    )
                    stats["stddev"] = variance**0.5
                elif "stddev" in operations:
                    stats["stddev"] = 0.0

            result = {
                "status": "success",
                "message": "Statistics calculated successfully.",
                "statistics": stats,
                "data_points_processed": count,
            }
        else:
            result = {
                "status": "failure",
                "error": "Statistical calculation failed (e.g., insufficient data, numerical instability).",
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result


class InternetInteractionTasks(BaseAgentTask):
    """Tasks related to interacting with internet resources like websites and APIs."""

    @staticmethod
    @add_tags("INET", "WEB")
    def fetch_url_content(url, timeout_seconds=10, method="GET", headers=None):
        """
        Simulates fetching content from a URL.

        Args:
            url (str): The URL to fetch.
            timeout_seconds (int, optional): Request timeout. Defaults to 10.
            method (str, optional): HTTP method ("GET", "POST", etc.). Defaults to "GET".
            headers (dict, optional): Custom headers.

        Returns:
            dict: Status, status code, and content.
                On success: {"status": "success", "status_code": int, "content": str, "headers": dict, "url": str}
                On failure: {"status": "failure", "status_code": int or None, "error": str, "url": str}
        """
        action_verb = f"fetch content from URL ({method})"
        print(f"Attempting to {action_verb}: {url} with timeout {timeout_seconds}s...")
        if _random_success_or_failure():
            status_code = random.choice([200, 201, 204])
            content_type = random.choice(
                ["text/html", "application/json", "text/plain"]
            )
            response_headers = {
                "Content-Type": content_type,
                "Content-Length": str(random.randint(100, 5000)),
            }
            if content_type == "text/html":
                content = f"<html><body><h1>Simulated Page: {url}</h1><p>Random content: {_generate_random_string(100)}</p></body></html>"
            elif content_type == "application/json":
                content = json.dumps(
                    {
                        "data": _generate_random_string(20),
                        "timestamp": datetime.datetime.utcnow().isoformat(),
                    }
                )
            else:
                content = f"Simulated plain text content from {url}.\n{_generate_random_string(200)}"

            result = {
                "status": "success",
                "status_code": status_code,
                "content": content,
                "headers": response_headers,
                "url": url,
                "method": method,
            }
        else:
            status_code = random.choice([400, 401, 403, 404, 500, 503])
            error_messages = {
                400: "Bad Request",
                401: "Unauthorized",
                403: "Forbidden",
                404: "URL Not Found",
                500: "Internal Server Error",
                503: "Service Unavailable",
            }
            result = {
                "status": "failure",
                "status_code": status_code,
                "error": error_messages.get(status_code, "Unknown HTTP error"),
                "url": url,
                "method": method,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("INET", "FM")
    def download_file_from_url(url, save_path, expected_file_type=None):
        """
        Simulates downloading a file from a URL and saving it locally.

        Args:
            url (str): The URL of the file to download.
            save_path (str): The local path where the file should be saved.
            expected_file_type (str, optional): E.g., "image/jpeg", "application/pdf". For simulation.

        Returns:
            dict: Status of the download.
                On success: {"status": "success", "message": "File downloaded", "file_url": str, "saved_to_path": str, "file_size_bytes": int}
                On failure: {"status": "failure", "error": str, "file_url": str}
        """
        action_verb = "download file"
        print(f"Attempting to {action_verb} from '{url}' to '{save_path}'...")
        if _random_success_or_failure():
            file_size = random.randint(1024, 1024 * 1024 * 5)  # 1KB to 5MB
            result = {
                "status": "success",
                "message": "File downloaded successfully.",
                "file_url": url,
                "saved_to_path": save_path,
                "file_size_bytes": file_size,
                "simulated_type": expected_file_type
                or random.choice(
                    ["application/octet-stream", "image/png", "application/zip"]
                ),
            }
        else:
            result = {
                "status": "failure",
                "error": "Failed to download file (e.g., URL not found, network error, disk full)",
                "file_url": url,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("INET", "API")
    def call_api_endpoint(
        endpoint_url, http_method="GET", headers=None, payload=None, auth_token=None
    ):
        """
        Simulates making a call to a generic API endpoint.

        Args:
            endpoint_url (str): The URL of the API endpoint.
            http_method (str, optional): HTTP method (GET, POST, PUT, DELETE). Defaults to "GET".
            headers (dict, optional): Custom request headers.
            payload (dict or str, optional): Data to send with POST/PUT requests.
            auth_token (str, optional): Bearer token or API key for authentication.

        Returns:
            dict: API response status.
                On success: {"status": "success", "status_code": int, "response_data": dict or list or str, "response_headers": dict}
                On failure: {"status": "failure", "status_code": int or None, "error": str, "error_details": dict or str}
        """
        action_verb = f"{http_method} request to API endpoint"
        print(f"Attempting {action_verb}: {endpoint_url}...")
        if _random_success_or_failure():
            status_code = random.choice([200, 201, 202, 204])
            response_data_type = random.choice(
                ["json_object", "json_array", "text", "empty"]
            )
            response_data = None
            if response_data_type == "json_object":
                response_data = {
                    "id": _generate_random_id(),
                    "value": _generate_random_string(),
                    "nested": {"key": "value"},
                }
            elif response_data_type == "json_array":
                response_data = [
                    {"id": _generate_random_id(), "name": _generate_random_string()}
                    for _ in range(random.randint(1, 5))
                ]
            elif response_data_type == "text":
                response_data = f"API Response: {_generate_random_string(50)}"

            result = {
                "status": "success",
                "message": "API call successful.",
                "status_code": status_code,
                "response_data": response_data,
                "response_headers": {
                    "Content-Type": (
                        "application/json"
                        if "json" in response_data_type
                        else "text/plain"
                    ),
                    "X-Request-ID": _generate_random_id("req_"),
                },
                "endpoint_url": endpoint_url,
                "method": http_method,
            }
        else:
            status_code = random.choice([400, 401, 403, 404, 429, 500, 503])
            error_messages = {
                400: "Bad Request (e.g. invalid payload)",
                401: "Unauthorized (e.g. missing/invalid token)",
                403: "Forbidden (insufficient permissions)",
                404: "Endpoint Not Found",
                429: "Too Many Requests (rate limited)",
                500: "Internal Server Error",
                503: "Service Unavailable",
            }
            result = {
                "status": "failure",
                "status_code": status_code,
                "error": error_messages.get(status_code, "Unknown API error"),
                "error_details": {
                    "code": f"ERR_{status_code}_{_generate_random_string(4)}",
                    "message": _get_random_error_message(action_verb),
                },
                "endpoint_url": endpoint_url,
                "method": http_method,
            }
        print(result)
        return result

    @staticmethod
    @add_tags("INET", "FM")
    def upload_file_to_server(local_file_path, server_url, upload_token=None):
        """
        Simulates uploading a local file to a server.

        Args:
            local_file_path (str): Path to the local file to upload.
            server_url (str): The URL of the server/endpoint to upload to.
            upload_token (str, optional): Authentication token for the upload.

        Returns:
            dict: Status of the upload.
                On success: {"status": "success", "message": "File uploaded", "file_path": str, "server_url": str, "remote_file_id": str}
                On failure: {"status": "failure", "error": str, "file_path": str}
        """
        action_verb = "upload file to server"
        print(f"Attempting to {action_verb} '{local_file_path}' to '{server_url}'...")
        if _random_success_or_failure():
            result = {
                "status": "success",
                "message": "File uploaded successfully.",
                "file_path": local_file_path,
                "server_url": server_url,
                "remote_file_id": _generate_random_id("remote_"),
                "upload_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
        else:
            result = {
                "status": "failure",
                "error": "Upload failed (e.g., timeout, server error, authentication failure, file too large)",
                "file_path": local_file_path,
                "server_url": server_url,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("INET", "WEB", "MONITOR")
    def check_website_status(url, expected_status_code=200):
        """
        Simulates checking the status of a website.

        Args:
            url (str): The URL of the website to check.
            expected_status_code (int, optional): The HTTP status code expected for success. Defaults to 200.

        Returns:
            dict: Website status.
                On success (reachable): {"status": "success", "message": "Website is reachable", "url": str, "http_status": int, "response_time_ms": float}
                On success (unreachable/wrong status): {"status": "success", "message": "Website check completed", "url": str, "http_status": int or None, "is_available": False}
                On failure (check process failed): {"status": "failure", "error": str, "url": str}
        """
        action_verb = "check website status"
        print(
            f"Attempting to {action_verb} for {url} (expecting {expected_status_code})..."
        )
        if _random_success_or_failure():  # The check operation itself succeeds
            is_actually_available = (
                random.random() > 0.15
            )  # 85% chance it's "available" in some form
            if is_actually_available:
                http_status = random.choice(
                    [200, 301, 302, 404, 500]
                )  # simulate various real statuses
                response_time = round(random.uniform(50, 2000), 2)  # ms
                result = {
                    "status": "success",  # The check itself was successful
                    "message": f"Website responded with status {http_status}.",
                    "url": url,
                    "is_available": http_status == expected_status_code,
                    "http_status_received": http_status,
                    "expected_http_status": expected_status_code,
                    "response_time_ms": response_time,
                }
            else:  # Simulates a timeout or DNS error from the website itself
                result = {
                    "status": "success",
                    "message": "Website check completed: site appears unreachable.",
                    "url": url,
                    "is_available": False,
                    "http_status_received": None,
                    "reason": random.choice(
                        [
                            "DNS resolution failed",
                            "Connection timed out",
                            "No response from server",
                        ]
                    ),
                }
        else:
            result = {
                "status": "failure",  # The checking tool/process itself failed
                "error": "Website status check failed (e.g., local network issue, checker tool error)",
                "url": url,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("INET", "SEARCH", "WEB")
    def perform_web_search(
        query_string, search_engine="simulated_google", num_results=5
    ):
        """
        Simulates performing a web search using a search engine.

        Args:
            query_string (str): The search query.
            search_engine (str, optional): Name of the search engine. Defaults to "simulated_google".
            num_results (int, optional): Number of results to simulate. Defaults to 5.

        Returns:
            dict: Search results.
                On success: {"status": "success", "query": str, "engine": str, "results": list[dict]}
                            (each dict: {"title": str, "url": str, "snippet": str})
                On failure: {"status": "failure", "error": str, "query": str}
        """
        action_verb = "perform web search"
        print(
            f"Attempting to {action_verb} for '{query_string}' on {search_engine} (requesting {num_results} results)..."
        )
        if _random_success_or_failure():
            results_list = []
            for i in range(num_results):
                domain = _generate_random_string(
                    random.randint(5, 10), "abcdefghijklmnopqrstuvwxyz"
                ) + random.choice([".com", ".org", ".net", ".io"])
                path = _generate_random_string(random.randint(5, 15))
                results_list.append(
                    {
                        "title": f"Simulated Result {i+1} for '{query_string[:20]}...' - {_generate_random_string(10).capitalize()}",
                        "url": f"https://www.{domain}/{path}",
                        "snippet": f"This is a simulated search result snippet for your query '{query_string}'. It contains relevant keywords like {_generate_random_string(8)} and {_generate_random_string(8)}...",
                        "rank": i + 1,
                    }
                )
            result = {
                "status": "success",
                "message": "Web search completed successfully.",
                "query": query_string,
                "engine": search_engine,
                "results_count": len(results_list),
                "results": results_list,
            }
        else:
            result = {
                "status": "failure",
                "error": "Web search failed (e.g., search engine API error, network connectivity issue, query blocked)",
                "query": query_string,
                "engine": search_engine,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result


class UtilityTasks(BaseAgentTask):
    """General utility tasks like random generation, conversions, logging."""

    @staticmethod
    @add_tags("UTIL", "RANDOM")
    def generate_random_number(min_val, max_val, number_type="integer"):
        """
        Simulates generating a random number within a specified range.

        Args:
            min_val (int or float): The minimum value (inclusive).
            max_val (int or float): The maximum value (inclusive).
            number_type (str, optional): "integer" or "float". Defaults to "integer".

        Returns:
            dict: Status and the generated number.
                On success: {"status": "success", "random_number": int or float, "range": [min, max], "type": str}
                On failure: {"status": "failure", "error": str}
        """
        action_verb = "generate random number"
        print(
            f"Attempting to {action_verb} between {min_val} and {max_val} (type: {number_type})..."
        )
        if _random_success_or_failure():
            try:
                if number_type == "integer":
                    if not (isinstance(min_val, int) and isinstance(max_val, int)):
                        raise ValueError(
                            "min_val and max_val must be integers for integer type."
                        )
                    number = random.randint(min_val, max_val)
                elif number_type == "float":
                    number = random.uniform(min_val, max_val)
                else:
                    raise ValueError(f"Unsupported number_type: {number_type}")

                result = {
                    "status": "success",
                    "random_number": number,
                    "range": [min_val, max_val],
                    "type": number_type,
                }
            except ValueError as e:
                result = {"status": "failure", "error": str(e)}

        else:
            result = {
                "status": "failure",
                "error": "Random number generation failed due to an unexpected system error.",
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("UTIL", "ID_GENERATION")
    def generate_unique_id(id_type="uuid"):
        """
        Simulates generating a unique identifier.

        Args:
            id_type (str, optional): Type of ID to generate ("uuid", "short_random", "timestamped"). Defaults to "uuid".

        Returns:
            dict: Status and the generated ID.
                On success: {"status": "success", "unique_id": str, "id_type": str}
                On failure: {"status": "failure", "error": str}
        """
        action_verb = "generate unique ID"
        print(f"Attempting to {action_verb} of type '{id_type}'...")
        if _random_success_or_failure():
            if id_type == "uuid":
                unique_id_val = str(uuid.uuid4())
            elif id_type == "short_random":
                unique_id_val = _generate_random_string(12)
            elif id_type == "timestamped":
                unique_id_val = f"{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{_generate_random_string(8)}"
            else:
                unique_id_val = (
                    f"unknown_type_{_generate_random_string(10)}"  # Fallback
                )

            result = {
                "status": "success",
                "unique_id": unique_id_val,
                "id_type": id_type,
            }
        else:
            result = {
                "status": "failure",
                "error": "Unique ID generation failed (e.g., entropy source issue).",
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("UTIL", "CONVERSION")
    def convert_units(value, from_unit, to_unit, category="temperature"):
        """
        Simulates converting a value from one unit to another (e.g., Celsius to Fahrenheit).

        Args:
            value (float): The numerical value to convert.
            from_unit (str): The unit to convert from (e.g., "C", "kg", "m").
            to_unit (str): The unit to convert to (e.g., "F", "lbs", "ft").
            category (str, optional): The category of conversion (e.g., "temperature", "weight", "length").

        Returns:
            dict: Status and converted value.
                On success: {"status": "success", "original_value": float, "from_unit": str,
                             "converted_value": float, "to_unit": str, "category": str}
                On failure: {"status": "failure", "error": str}
        """
        action_verb = f"convert units ({category})"
        print(f"Attempting to {action_verb}: {value} {from_unit} to {to_unit}...")
        if _random_success_or_failure():
            # Simplified simulation: apply a random factor or a known simple conversion
            converted_value = value * random.uniform(0.5, 2.0)  # Generic placeholder
            if category == "temperature" and from_unit == "C" and to_unit == "F":
                converted_value = (value * 9 / 5) + 32
            elif category == "temperature" and from_unit == "F" and to_unit == "C":
                converted_value = (value - 32) * 5 / 9
            elif category == "length" and from_unit == "m" and to_unit == "ft":
                converted_value = value * 3.28084
            elif category == "length" and from_unit == "ft" and to_unit == "m":
                converted_value = value / 3.28084

            result = {
                "status": "success",
                "message": "Unit conversion successful.",
                "original_value": value,
                "from_unit": from_unit,
                "converted_value": round(converted_value, 4),
                "to_unit": to_unit,
                "category": category,
            }
        else:
            result = {
                "status": "failure",
                "error": f"Unit conversion failed (e.g., unknown units '{from_unit}'/'{to_unit}', incompatible categories).",
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("UTIL", "DATETIME")
    def get_current_timestamp(timezone="UTC", format_string="%Y-%m-%dT%H:%M:%S%z"):
        """
        Simulates getting the current date and time.

        Args:
            timezone (str, optional): The timezone (e.g., "UTC", "America/New_York"). Simulation may not fully support.
            format_string (str, optional): Desired output format string for strftime.

        Returns:
            dict: Status and current timestamp.
                On success: {"status": "success", "timestamp_str": str, "timezone_used": str, "epoch_seconds": float}
                On failure: {"status": "failure", "error": str}
        """
        action_verb = "get current timestamp"
        print(
            f"Attempting to {action_verb} (timezone: {timezone}, format: '{format_string}')..."
        )
        if _random_success_or_failure():
            # Simulate timezone awareness minimally
            now = (
                datetime.datetime.utcnow()
                if timezone == "UTC"
                else datetime.datetime.now()
            )
            try:
                ts_str = now.strftime(format_string)
                # Add 'Z' for UTC if format string doesn't handle it well for naive UTC.
                if (
                    timezone == "UTC"
                    and format_string.endswith("%z")
                    and not ts_str.endswith("0000")
                ):
                    if not (
                        ts_str.endswith("Z") or "+" in ts_str[-5:] or "-" in ts_str[-5:]
                    ):  # crude check
                        ts_str = now.strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        )  # Force ISO with Z for UTC
            except ValueError:  # Invalid format string
                ts_str = now.isoformat()  # Fallback to ISO

            result = {
                "status": "success",
                "timestamp_str": ts_str,
                "timezone_used": timezone,  # Simplified, real tz handling is complex
                "epoch_seconds": now.timestamp(),
                "iso_format": now.isoformat(),
            }
        else:
            result = {
                "status": "failure",
                "error": "Failed to retrieve current timestamp (e.g., system clock error).",
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("UTIL", "CALCULATION")
    def perform_basic_calculation(expression_string):
        """
        Simulates performing a basic arithmetic calculation from a string expression.
        WARNING: Real eval() is dangerous. This is a highly restricted simulation.

        Args:
            expression_string (str): The arithmetic expression (e.g., "2 + 2 * 3").

        Returns:
            dict: Status and calculation result.
                On success: {"status": "success", "expression": str, "result": float or int}
                On failure: {"status": "failure", "error": str, "expression": str}
        """
        action_verb = "perform basic calculation"
        print(f"Attempting to {action_verb} for expression: '{expression_string}'...")
        # Extremely simple and "safe" evaluation for simulation
        allowed_chars = set("0123456789.+-*/() ")
        if not all(char in allowed_chars for char in expression_string):
            result = {
                "status": "failure",
                "error": "Invalid characters in expression.",
                "expression": expression_string,
            }
        elif _random_success_or_failure():
            try:
                # In a real scenario, use a proper math expression parser, NOT eval()
                # For simulation, we'll try a very limited eval
                if any(
                    op in expression_string for op in ["**", "//"]
                ):  # Disallow complex ops
                    raise ValueError("Complex operators not allowed in simulation.")
                # Simple check for safety, not foolproof
                if len(expression_string) > 50:
                    raise ValueError("Expression too long.")

                # A slightly safer "simulated" eval
                simulated_result = None
                if "+" in expression_string and len(expression_string.split("+")) == 2:
                    parts = expression_string.split("+")
                    simulated_result = float(parts[0].strip()) + float(parts[1].strip())
                elif (
                    "*" in expression_string and len(expression_string.split("*")) == 2
                ):
                    parts = expression_string.split("*")
                    simulated_result = float(parts[0].strip()) * float(parts[1].strip())
                else:  # Fallback, very basic
                    simulated_result = (
                        random.uniform(0, 100)
                        if "." in expression_string
                        else random.randint(0, 100)
                    )

                result = {
                    "status": "success",
                    "expression": expression_string,
                    "result": simulated_result,
                    "message": "Calculation performed (simulated evaluation).",
                }
            except Exception as e:  # Catch errors from float conversion or other issues
                result = {
                    "status": "failure",
                    "error": f"Error during simulated calculation: {str(e)}",
                    "expression": expression_string,
                }
        else:
            result = {
                "status": "failure",
                "error": "Calculation engine failed or expression is too complex/invalid.",
                "expression": expression_string,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("UTIL", "LOGGING")
    def log_custom_event(log_level, message, component_name, additional_data=None):
        """
        Simulates logging a custom event or message.

        Args:
            log_level (str): Severity level (e.g., "INFO", "WARNING", "ERROR", "DEBUG").
            message (str): The main log message.
            component_name (str): Name of the component/module logging the event.
            additional_data (dict, optional): Extra key-value pairs for structured logging.

        Returns:
            dict: Status of the logging operation.
                On success: {"status": "success", "message": "Event logged", "log_id": str}
                On failure: {"status": "failure", "error": str}
        """
        action_verb = "log custom event"
        print(
            f"Attempting to {action_verb} (Level: {log_level}, Component: {component_name}): {message[:50]}..."
        )
        if _random_success_or_failure():
            result = {
                "status": "success",
                "message": "Event logged successfully to simulated logging system.",
                "log_id": _generate_random_id("log_"),
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "details_logged": {
                    "level": log_level,
                    "message": message,
                    "component": component_name,
                    "additional_data": additional_data or {},
                },
            }
        else:
            result = {
                "status": "failure",
                "error": "Logging system unavailable or write error.",
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result


class DatabaseInteractionTasks(BaseAgentTask):
    """Tasks related to interacting with databases."""

    @staticmethod
    @add_tags("DB", "QUERY")
    def query_database(
        database_name, query_string, parameters=None, query_type="SELECT"
    ):
        """
        Simulates executing a query against a database.

        Args:
            database_name (str): Name or connection identifier of the database.
            query_string (str): The SQL-like query string.
            parameters (dict or list, optional): Parameters for prepared statements.
            query_type (str, optional): Type of query ("SELECT", "INSERT", "UPDATE", "DELETE").

        Returns:
            dict: Query result.
                On success (SELECT): {"status": "success", "records": list[dict], "row_count": int}
                On success (INSERT/UPDATE/DELETE): {"status": "success", "rows_affected": int}
                On failure: {"status": "failure", "error": str, "database_name": str}
        """
        action_verb = f"execute {query_type} query on database"
        print(f"Attempting to {action_verb} '{database_name}': {query_string[:100]}...")
        if _random_success_or_failure():
            if query_type.upper() == "SELECT":
                num_records = random.randint(0, 10)
                records = [
                    {
                        "id": i + 1,
                        "name": _generate_random_string(),
                        "value": random.random() * 100,
                    }
                    for i in range(num_records)
                ]
                result = {
                    "status": "success",
                    "message": "Query executed successfully.",
                    "records": records,
                    "row_count": num_records,
                    "columns_returned": (
                        ["id", "name", "value"] if num_records > 0 else []
                    ),
                }
            else:  # INSERT, UPDATE, DELETE
                rows_affected = (
                    random.randint(0, 5)
                    if query_type.upper() == "UPDATE" or query_type.upper() == "DELETE"
                    else 1
                )
                result = {
                    "status": "success",
                    "message": f"{query_type.capitalize()} operation successful.",
                    "rows_affected": rows_affected,
                }
            result["database_name"] = database_name
            result["query_executed"] = query_string
        else:
            result = {
                "status": "failure",
                "error": f"Database query failed (e.g., connection error, syntax error, constraint violation, '{database_name}' not reachable).",
                "database_name": database_name,
                "query_attempted": query_string,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("DB", "WRITE")
    def insert_database_record(database_name, table_name, record_data):
        """
        Simulates inserting a new record into a database table.

        Args:
            database_name (str): Name of the database.
            table_name (str): Name of the table.
            record_data (dict): Key-value pairs representing the record to insert.

        Returns:
            dict: Status of the insert operation.
                On success: {"status": "success", "message": "Record inserted", "record_id": any, "table_name": str}
                On failure: {"status": "failure", "error": str, "table_name": str}
        """
        action_verb = "insert database record"
        print(f"Attempting to {action_verb} into {database_name}.{table_name}...")
        if _random_success_or_failure():
            inserted_id = (
                _generate_random_id("rec_")
                if random.random() > 0.1
                else random.randint(1000, 9999)
            )
            result = {
                "status": "success",
                "message": "Record inserted successfully.",
                "record_id": inserted_id,
                "database_name": database_name,
                "table_name": table_name,
                "data_inserted_preview": {
                    k: str(v)[:20] for k, v in record_data.items()
                },
            }
        else:
            result = {
                "status": "failure",
                "error": "Failed to insert record (e.g., unique constraint violation, invalid data type, table lock).",
                "database_name": database_name,
                "table_name": table_name,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("DB", "WRITE")
    def update_database_record(database_name, table_name, record_id, update_data):
        """
        Simulates updating an existing record in a database table.

        Args:
            database_name (str): Name of the database.
            table_name (str): Name of the table.
            record_id (any): The ID of the record to update.
            update_data (dict): Key-value pairs representing the fields to update.

        Returns:
            dict: Status of the update operation.
                On success: {"status": "success", "message": "Record updated", "rows_affected": int, "record_id": any}
                On failure: {"status": "failure", "error": str, "record_id": any}
        """
        action_verb = "update database record"
        print(
            f"Attempting to {action_verb} ID {record_id} in {database_name}.{table_name}..."
        )
        if _random_success_or_failure():
            # Simulate if record was found and updated or not found
            rows_affected = (
                1 if random.random() > 0.15 else 0
            )  # 85% chance record "exists"
            message = (
                "Record updated successfully."
                if rows_affected > 0
                else "Record found but no changes made or record not found."
            )
            result = {
                "status": "success",
                "message": message,
                "rows_affected": rows_affected,
                "record_id": record_id,
                "database_name": database_name,
                "table_name": table_name,
            }
        else:
            result = {
                "status": "failure",
                "error": "Failed to update record (e.g., record not found, concurrency conflict, invalid data).",
                "record_id": record_id,
                "database_name": database_name,
                "table_name": table_name,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("DB", "WRITE")
    def delete_database_record(database_name, table_name, record_id):
        """
        Simulates deleting a record from a database table.

        Args:
            database_name (str): Name of the database.
            table_name (str): Name of the table.
            record_id (any): The ID of the record to delete.

        Returns:
            dict: Status of the delete operation.
                On success: {"status": "success", "message": "Record deleted", "rows_affected": int, "record_id": any}
                On failure: {"status": "failure", "error": str, "record_id": any}
        """
        action_verb = "delete database record"
        print(
            f"Attempting to {action_verb} ID {record_id} from {database_name}.{table_name}..."
        )
        if _random_success_or_failure():
            rows_affected = (
                1 if random.random() > 0.2 else 0
            )  # 80% chance record "existed" to be deleted
            message = (
                "Record deleted successfully."
                if rows_affected > 0
                else "Record not found or already deleted."
            )
            result = {
                "status": "success",
                "message": message,
                "rows_affected": rows_affected,
                "record_id": record_id,
                "database_name": database_name,
                "table_name": table_name,
            }
        else:
            result = {
                "status": "failure",
                "error": "Failed to delete record (e.g., foreign key constraint, record locked).",
                "record_id": record_id,
                "database_name": database_name,
                "table_name": table_name,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result


class SystemControlTasks(BaseAgentTask):
    """Tasks related to simulated system control and monitoring."""

    @staticmethod
    @add_tags("SYS", "MONITOR")
    def check_system_resource_usage(resource_type="cpu", target_system="localhost"):
        """
        Simulates checking system resource usage (CPU, memory, disk).

        Args:
            resource_type (str, optional): "cpu", "memory", or "disk". Defaults to "cpu".
            target_system (str, optional): Identifier of the system to check. Defaults to "localhost".

        Returns:
            dict: Status and resource usage details.
                On success: {"status": "success", "resource_type": str, "usage_percent": float, "details": dict}
                On failure: {"status": "failure", "error": str, "resource_type": str}
        """
        action_verb = f"check {resource_type} usage on {target_system}"
        print(f"Attempting to {action_verb}...")
        if _random_success_or_failure():
            usage_percent = round(random.uniform(5.0, 95.0), 2)
            details = {}
            if resource_type == "cpu":
                details = {
                    "load_average_1m": round(random.uniform(0.1, 4.0), 2),
                    "cores": random.randint(1, 16),
                }
            elif resource_type == "memory":
                total_gb = random.choice([4, 8, 16, 32])
                used_gb = round((usage_percent / 100) * total_gb, 2)
                details = {
                    "total_gb": total_gb,
                    "used_gb": used_gb,
                    "free_gb": round(total_gb - used_gb, 2),
                }
            elif resource_type == "disk":
                total_gb_disk = random.choice([100, 256, 512, 1024])
                used_gb_disk = round((usage_percent / 100) * total_gb_disk, 2)
                details = {
                    "path": "/",
                    "total_gb": total_gb_disk,
                    "used_gb": used_gb_disk,
                    "free_gb": round(total_gb_disk - used_gb_disk, 2),
                }

            result = {
                "status": "success",
                "message": f"{resource_type.capitalize()} usage checked successfully.",
                "target_system": target_system,
                "resource_type": resource_type,
                "usage_percent": usage_percent,
                "details": details,
            }
        else:
            result = {
                "status": "failure",
                "error": f"Failed to check {resource_type} usage (e.g., monitoring agent down, permissions error).",
                "target_system": target_system,
                "resource_type": resource_type,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("SYS", "EXECUTE")
    def execute_shell_command(
        command_string, target_system="localhost", timeout_seconds=30
    ):
        """
        Simulates executing a shell command.
        WARNING: Real execution is highly dangerous. This is purely a simulation.

        Args:
            command_string (str): The command to "execute".
            target_system (str, optional): System where command would run. Defaults to "localhost".
            timeout_seconds (int, optional): Simulated timeout. Defaults to 30.

        Returns:
            dict: Simulated command output.
                On success: {"status": "success", "command": str, "exit_code": 0, "stdout": str, "stderr": str}
                On failure: {"status": "failure", "command": str, "exit_code": int (non-zero), "error": str, "stderr": str}
        """
        action_verb = f"execute shell command on {target_system}"
        print(
            f"Attempting to (SIMULATE) {action_verb}: '{command_string}' with timeout {timeout_seconds}s..."
        )
        # Sanitize command for safety in simulation print/log
        safe_command = command_string.replace("`", "'").replace("$", "")[:100]

        if _random_success_or_failure():
            # Simulate common command outputs
            stdout_content = ""
            if "ls" in command_string or "dir" in command_string:
                stdout_content = "file1.txt\ndirectoryA\nscript.sh"
            elif "echo" in command_string:
                stdout_content = command_string.split("echo", 1)[-1].strip()
            elif "pwd" in command_string:
                stdout_content = "/simulated/current/directory"
            else:
                stdout_content = f"Simulated output for command: {safe_command}"

            result = {
                "status": "success",
                "message": "Shell command simulated successfully.",
                "command": safe_command,
                "target_system": target_system,
                "exit_code": 0,
                "stdout": stdout_content,
                "stderr": "",  # Assume no error output on success
            }
        else:
            exit_code = random.randint(1, 127)
            stderr_content = random.choice(
                [
                    f"Error: command not found: {safe_command.split(' ')[0]}",
                    "Permission denied.",
                    f"Timeout after {timeout_seconds} seconds.",
                    "An unexpected error occurred during execution.",
                ]
            )
            result = {
                "status": "failure",
                "command": safe_command,
                "target_system": target_system,
                "exit_code": exit_code,
                "error": "Simulated command execution failed.",
                "stdout": "",  # Assume no stdout on failure
                "stderr": stderr_content,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("SYS", "SCHEDULE")
    def schedule_system_task(
        task_name, command_to_run, schedule_expression, task_type="cron"
    ):
        """
        Simulates scheduling a recurring system task (e.g., a cron job).

        Args:
            task_name (str): A name for the scheduled task.
            command_to_run (str): The command or script to be executed.
            schedule_expression (str): The schedule (e.g., cron string "0 5 * * *").
            task_type (str, optional): Type of scheduler ("cron", "systemd_timer"). Defaults to "cron".

        Returns:
            dict: Status of the scheduling operation.
                On success: {"status": "success", "task_id": str, "task_name": str, "schedule": str}
                On failure: {"status": "failure", "error": str, "task_name": str}
        """
        action_verb = "schedule system task"
        print(
            f"Attempting to {action_verb} '{task_name}' with schedule '{schedule_expression}' ({task_type})..."
        )
        if _random_success_or_failure():
            result = {
                "status": "success",
                "message": "System task scheduled successfully.",
                "task_id": _generate_random_id(f"{task_type}_task_"),
                "task_name": task_name,
                "command": command_to_run,
                "schedule_expression": schedule_expression,
                "task_type": task_type,
            }
        else:
            result = {
                "status": "failure",
                "error": "Failed to schedule system task (e.g., invalid schedule format, permissions error, scheduler service down).",
                "task_name": task_name,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("SYS", "SERVICE_MGMT")
    def manage_system_service(service_name, action, target_system="localhost"):
        """
        Simulates managing a system service (start, stop, restart, status).

        Args:
            service_name (str): Name of the system service (e.g., "nginx", "sshd").
            action (str): "start", "stop", "restart", or "status".
            target_system (str, optional): System where service runs. Defaults to "localhost".

        Returns:
            dict: Status of the service management operation.
                On success (start/stop/restart): {"status": "success", "message": "Service action completed", "service_name": str, "action": str}
                On success (status): {"status": "success", "service_name": str, "service_status": "running"|"stopped"|"unknown"}
                On failure: {"status": "failure", "error": str, "service_name": str, "action": str}
        """
        full_action_verb = f"{action} service '{service_name}' on {target_system}"
        print(f"Attempting to {full_action_verb}...")
        if action not in ["start", "stop", "restart", "status"]:
            return {
                "status": "failure",
                "error": "Invalid action. Must be 'start', 'stop', 'restart', or 'status'.",
                "service_name": service_name,
                "action": action,
            }

        if _random_success_or_failure():
            if action == "status":
                service_status = random.choice(
                    ["running", "stopped", "unknown", "failed"]
                )
                result = {
                    "status": "success",
                    "message": f"Status for service '{service_name}' retrieved.",
                    "service_name": service_name,
                    "action": action,
                    "target_system": target_system,
                    "service_status": service_status,
                    "details": {
                        "pid": (
                            random.randint(1000, 30000)
                            if service_status == "running"
                            else None
                        ),
                        "uptime_seconds": (
                            random.randint(60, 86400)
                            if service_status == "running"
                            else 0
                        ),
                    },
                }
            else:  # start, stop, restart
                result = {
                    "status": "success",
                    "message": f"Service '{service_name}' action '{action}' completed successfully.",
                    "service_name": service_name,
                    "action": action,
                    "target_system": target_system,
                }
        else:
            result = {
                "status": "failure",
                "error": f"Failed to {action} service '{service_name}' (e.g., service not found, permissions error, timeout).",
                "service_name": service_name,
                "action": action,
                "target_system": target_system,
                "details": _get_random_error_message(full_action_verb),
            }
        print(result)
        return result


class UserInteractionTasks(BaseAgentTask):
    """Tasks simulating interactions with a user (e.g., prompts, notifications)."""

    @staticmethod
    @add_tags("UI", "PROMPT")
    def request_user_confirmation(
        prompt_message, options=["Yes", "No"], default_option="No"
    ):
        """
        Simulates requesting a yes/no confirmation from the user.

        Args:
            prompt_message (str): The message/question to display to the user.
            options (list[str], optional): List of confirmation options. Defaults to ["Yes", "No"].
            default_option (str, optional): The default choice if user times out (simulated).

        Returns:
            dict: User's response.
                On success: {"status": "success", "prompt": str, "response": str (e.g. "Yes"), "confirmed": bool}
                On failure (simulation context): {"status": "failure", "error": "User interaction module unavailable"}
        """
        action_verb = "request user confirmation"
        print(
            f"Attempting to (SIMULATE) {action_verb}: '{prompt_message}' Options: {options}"
        )
        if _random_success_or_failure():  # Simulates UI being available
            # Simulate user choosing an option
            user_choice = (
                random.choice(options) if random.random() > 0.1 else default_option
            )  # 10% chance of "timeout" to default
            is_confirmed = (
                user_choice.lower() == "yes"
                or user_choice.lower() == options[0].lower()
            )  # Crude positive confirmation check

            result = {
                "status": "success",
                "message": "User confirmation received.",
                "prompt": prompt_message,
                "response": user_choice,
                "confirmed": is_confirmed,
                "options_provided": options,
            }
        else:
            result = {
                "status": "failure",
                "error": "User interaction module unavailable or timed out.",
                "prompt": prompt_message,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("UI", "DISPLAY")
    def display_notification_to_user(
        message_text, message_type="info", duration_seconds=5
    ):
        """
        Simulates displaying a non-blocking notification message to the user.

        Args:
            message_text (str): The content of the notification.
            message_type (str, optional): "info", "warning", "error", "success". Defaults to "info".
            duration_seconds (int, optional): How long the notification might be displayed.

        Returns:
            dict: Status of the notification display.
                On success: {"status": "success", "message": "Notification displayed", "type": str, "notification_id": str}
                On failure: {"status": "failure", "error": "Notification system error"}
        """
        action_verb = "display notification to user"
        print(
            f"Attempting to (SIMULATE) {action_verb} (Type: {message_type}): '{message_text[:50]}...'"
        )
        if _random_success_or_failure():
            result = {
                "status": "success",
                "message": "Notification displayed to user (simulated).",
                "notification_id": _generate_random_id("notify_"),
                "type": message_type,
                "text_displayed": message_text,
                "duration_seconds": duration_seconds,
            }
        else:
            result = {
                "status": "failure",
                "error": "Notification system error or user UI not available.",
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("UI", "INPUT")
    def prompt_user_for_input(
        prompt_message, input_type="text", default_value=None, validation_regex=None
    ):
        """
        Simulates prompting the user for textual input.

        Args:
            prompt_message (str): The message asking for input.
            input_type (str, optional): "text", "password", "number". Defaults to "text".
            default_value (str, optional): A default value for the input field.
            validation_regex (str, optional): A regex string for simulated validation.

        Returns:
            dict: User's input.
                On success: {"status": "success", "prompt": str, "input_value": str, "input_type": str}
                On success (empty/cancelled): {"status": "success", "prompt": str, "input_value": None, "message": "User cancelled or provided no input"}
                On failure: {"status": "failure", "error": "User input module unavailable"}
        """
        action_verb = "prompt user for input"
        print(
            f"Attempting to (SIMULATE) {action_verb}: '{prompt_message}' (Type: {input_type})"
        )
        if _random_success_or_failure():
            # Simulate user providing input or cancelling
            if random.random() < 0.15:  # 15% chance user cancels or provides no input
                user_input = None
                msg = "User cancelled or provided no input."
            else:
                if input_type == "text":
                    user_input = (
                        _generate_random_string(random.randint(5, 20))
                        + " "
                        + _generate_random_string(random.randint(3, 10))
                    )
                elif input_type == "password":
                    user_input = "********"  # Masked
                elif input_type == "number":
                    user_input = str(random.randint(0, 1000))
                else:
                    user_input = _generate_random_string(10)
                msg = "User input received."

            result = {
                "status": "success",
                "message": msg,
                "prompt": prompt_message,
                "input_value": user_input,  # In real password case, never log/return actual password
                "input_type": input_type,
                "default_provided": default_value,
            }
        else:
            result = {
                "status": "failure",
                "error": "User input module unavailable or timed out.",
                "prompt": prompt_message,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result


class AuthenticationTasks(BaseAgentTask):
    """Tasks related to user authentication and authorization."""

    @staticmethod
    @add_tags("AUTH", "SECURITY")
    def authenticate_user_credentials(
        username, password_hash_or_actual, auth_system="local_db"
    ):
        """
        Simulates authenticating a user with username and password (or hash).
        Never use actual passwords in real systems like this.

        Args:
            username (str): The username.
            password_hash_or_actual (str): The user's password (for simulation) or its hash.
            auth_system (str, optional): System used for auth (e.g., "local_db", "LDAP", "OAuth").

        Returns:
            dict: Authentication status.
                On success: {"status": "success", "message": "Authentication successful", "user_id": str, "session_token": str, "roles": list}
                On failure (bad creds): {"status": "failure", "error": "Invalid credentials", "username": str}
                On failure (system error): {"status": "failure", "error": "Authentication system error", "username": str}
        """
        action_verb = f"authenticate user credentials via {auth_system}"
        print(f"Attempting to {action_verb} for user '{username}'...")
        # This simulates the auth system itself working or not
        auth_system_operational = _random_success_or_failure(
            success_rate=0.95
        )  # High chance auth system is OK

        if auth_system_operational:
            # Now simulate if credentials are valid (independent of system operational status)
            credentials_valid = _random_success_or_failure(
                success_rate=0.7
            )  # 70% chance creds are "valid"
            if credentials_valid:
                result = {
                    "status": "success",
                    "message": "Authentication successful.",
                    "user_id": _generate_random_id(f"{username}_"),
                    "username": username,
                    "session_token": _generate_random_token(),
                    "token_expires_in_seconds": 3600,  # 1 hour
                    "roles": random.sample(
                        ["user", "editor", "viewer", "admin_basic"],
                        k=random.randint(1, 2),
                    ),
                    "auth_system": auth_system,
                }
            else:
                result = {
                    "status": "failure",
                    "error": "Invalid credentials (username or password incorrect).",
                    "username": username,
                    "auth_system": auth_system,
                    "reason_code": "AUTH_INVALID_CREDENTIALS",
                }
        else:
            result = {
                "status": "failure",
                "error": f"Authentication system '{auth_system}' unavailable or encountered an error.",
                "username": username,
                "auth_system": auth_system,
                "reason_code": "AUTH_SYSTEM_ERROR",
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result

    @staticmethod
    @add_tags("AUTH", "SECURITY")
    def validate_session_token(session_token, required_permission=None):
        """
        Simulates validating a user's session token and optionally checking a permission.

        Args:
            session_token (str): The session token to validate.
            required_permission (str, optional): A specific permission string to check for (e.g., "file:write").

        Returns:
            dict: Token validation status.
                On success (valid): {"status": "success", "is_valid": True, "user_id": str, "permissions_granted": list}
                On success (invalid/expired): {"status": "success", "is_valid": False, "reason": str}
                On failure (validation system error): {"status": "failure", "error": str}
        """
        action_verb = "validate session token"
        print(
            f"Attempting to {action_verb} '{session_token[:15]}...' (Permission required: {required_permission})..."
        )
        validation_system_ok = _random_success_or_failure(
            0.98
        )  # System itself is likely fine

        if validation_system_ok:
            token_actually_valid = _random_success_or_failure(
                0.8
            )  # 80% chance token is "valid"
            if token_actually_valid:
                user_permissions = random.sample(
                    [
                        "file:read",
                        "file:write",
                        "email:send",
                        "user:manage",
                        "report:generate",
                    ],
                    k=random.randint(1, 4),
                )
                permission_met = True
                if required_permission and required_permission not in user_permissions:
                    permission_met = False

                if permission_met:
                    result = {
                        "status": "success",
                        "message": "Session token is valid and permission requirements met.",
                        "is_valid": True,
                        "user_id": _generate_random_id("user_"),
                        "permissions_granted": user_permissions,
                        "token_expires_at_iso": (
                            datetime.datetime.utcnow()
                            + datetime.timedelta(minutes=random.randint(5, 59))
                        ).isoformat()
                        + "Z",
                    }
                else:
                    result = {
                        "status": "success",  # Validation check completed
                        "message": "Session token is valid, but required permission not granted.",
                        "is_valid": True,  # Token itself is valid
                        "permission_check_failed": True,
                        "required_permission": required_permission,
                        "user_permissions": user_permissions,
                    }
            else:  # Token is invalid or expired
                reason = random.choice(
                    [
                        "Token expired",
                        "Token revoked",
                        "Invalid token format",
                        "User session terminated",
                    ]
                )
                result = {
                    "status": "success",  # Validation check completed
                    "message": f"Session token is invalid: {reason}",
                    "is_valid": False,
                    "reason": reason,
                    "reason_code": f"TOKEN_INVALID_{reason.upper().replace(' ','_')}",
                }
        else:
            result = {
                "status": "failure",
                "error": "Token validation service unavailable or encountered an internal error.",
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result


class OrchestrationTasks(BaseAgentTask):
    """Tasks related to delegating work or coordinating multiple steps/agents."""

    @staticmethod
    @add_tags("ORCH", "DELEGATE")
    def delegate_task_to_another_agent(
        target_agent_id, task_name, task_parameters, priority="normal"
    ):
        """
        Simulates delegating a specific task to another agent.

        Args:
            target_agent_id (str): Identifier of the agent to delegate to.
            task_name (str): Name or description of the task to be performed.
            task_parameters (dict): Parameters required for the task.
            priority (str, optional): "high", "normal", "low". Defaults to "normal".

        Returns:
            dict: Status of the delegation attempt.
                On success: {"status": "success", "message": "Task delegated", "delegation_id": str, "target_agent": str}
                On failure: {"status": "failure", "error": str, "target_agent": str}
        """
        action_verb = "delegate task to another agent"
        print(
            f"Attempting to {action_verb} '{task_name}' to agent '{target_agent_id}' with priority {priority}..."
        )
        if _random_success_or_failure():
            result = {
                "status": "success",
                "message": f"Task '{task_name}' successfully delegated to agent '{target_agent_id}'.",
                "delegation_id": _generate_random_id("dlg_"),
                "target_agent_id": target_agent_id,
                "task_name_delegated": task_name,
                "task_parameters_sent": task_parameters,
                "priority_set": priority,
                "expected_completion_estimate_minutes": random.randint(
                    5, 120
                ),  # Simulated
            }
        else:
            result = {
                "status": "failure",
                "error": f"Failed to delegate task to agent '{target_agent_id}' (e.g., agent offline, task queue full, invalid task).",
                "target_agent_id": target_agent_id,
                "task_name_attempted": task_name,
                "details": _get_random_error_message(action_verb),
            }
        print(result)
        return result


# --- Collect functions by category ---
# List of all task category classes defined above
ALL_TASK_CLASSES = [
    CommunicationTasks,
    FileManagementTasks,
    DataProcessingTasks,
    InternetInteractionTasks,
    UtilityTasks,
    DatabaseInteractionTasks,
    SystemControlTasks,
    UserInteractionTasks,
    AuthenticationTasks,
    OrchestrationTasks,
]

# This dictionary will hold functions categorized by their class name
# Format: {"CategoryName": [function_ref1, function_ref2, ...]}
categorized_functions = {}

# This dictionary will hold more details, including tags
# Format: {"CategoryName": [{"name": "FunctionName", "function": func_ref, "tags": ["TAG1", "TAG2"]}, ...]}
categorized_functions_with_details = {}

for task_class in ALL_TASK_CLASSES:
    # Derive category name from class name (e.g., "CommunicationTasks" -> "Communication")
    category_name = task_class.__name__.replace("Tasks", "")

    functions_in_class = []
    functions_details_in_class = []

    # Iterate over members of the class
    for name, member in inspect.getmembers(task_class):
        # Check if it's a static method (which is a function) and not a private/helper method
        if (
            inspect.isfunction(member)
            and not name.startswith("_")
            and hasattr(task_class, name)
        ):
            # Ensure it's a method defined in this class (not inherited from BaseAgentTask if any non-static methods were there)
            # For static methods, this check is a bit more nuanced, but generally works if BaseAgentTask has no conflicting static method names.
            # A more robust check might involve checking `member.__qualname__`.

            # Add to the simple list of function references
            functions_in_class.append(member)

            # Populate tags: start with the primary category (derived from class name)
            # and add any tags defined by the @add_tags decorator.
            primary_tag = category_name.upper()
            additional_tags = getattr(member, "tags", [])
            all_tags = sorted(
                list(set([primary_tag] + [t.upper() for t in additional_tags]))
            )

            # Store detailed info
            functions_details_in_class.append(
                {
                    "name": f"{task_class.__name__}.{name}",  # Fully qualified-like name
                    "function": member,
                    "docstring": inspect.getdoc(member),
                    "parameters": list(inspect.signature(member).parameters.keys()),
                    "tags": all_tags,
                }
            )

            # Also, assign the final computed tags back to the function object itself for easy access
            member.computed_tags = all_tags

    if functions_in_class:
        categorized_functions[category_name] = functions_in_class
        categorized_functions_with_details[category_name] = functions_details_in_class


if __name__ == "__main__":
    print("\n\n--- DEMONSTRATION OF CATEGORIZED FUNCTIONS ---")

    # Print the categories and the number of functions in each
    print("\nFunction Counts per Category:")
    for category, funcs in categorized_functions.items():
        print(f"- {category}: {len(funcs)} functions")

    # Example of accessing and calling a function
    print("\nExample: Calling 'send_email' from CommunicationTasks")
    if (
        "Communication" in categorized_functions
        and CommunicationTasks.send_email in categorized_functions["Communication"]
    ):
        email_func = CommunicationTasks.send_email
        result = email_func(
            sender_email="agent@example.com",
            recipient_emails=["user1@example.com", "user2@example.com"],
            subject="AI Agent Test Email",
            body="This is a test email sent by the AI agent simulation.",
        )
        print(f"Result of calling {email_func.__name__}: {result['status']}")
        print(
            f"Tags for {email_func.__name__}: {getattr(email_func, 'computed_tags', 'N/A')}"
        )

    print("\nExample: Calling 'read_file_content' from FileManagementTasks")
    if (
        "FileManagement" in categorized_functions
        and FileManagementTasks.read_file_content
        in categorized_functions["FileManagement"]
    ):
        read_func = FileManagementTasks.read_file_content
        result = read_func(file_path="/simulated/path/to/document.txt")
        print(f"Result of calling {read_func.__name__}: {result['status']}")
        if result["status"] == "success":
            print(f"  Content preview: {result['content'][:50]}...")
        print(
            f"Tags for {read_func.__name__}: {getattr(read_func, 'computed_tags', 'N/A')}"
        )

    print("\nListing all functions with their details (first few per category):")
    for category, func_details_list in categorized_functions_with_details.items():
        print(f"\nCategory: {category}")
        for i, func_detail in enumerate(func_details_list):
            if i < 3:  # Print details for first 3 functions per category for brevity
                print(f"  Function: {func_detail['name']}")
                print(f"    Tags: {func_detail['tags']}")
                print(f"    Parameters: {func_detail['parameters']}")
                # print(f"    Docstring: {func_detail['docstring'][:60]}...") # Uncomment for docstring preview
            elif i == 3:
                print("    ... and more.")
                break

    total_functions = sum(len(funcs) for funcs in categorized_functions.values())
    print(f"\nTotal stub functions generated: {total_functions}")

    # You can now use `categorized_functions` or `categorized_functions_with_details`
    # for your AI agent training or other purposes. For example:
    # agent_tool_registry = categorized_functions

import random
from collections import defaultdict
from typing import List, Dict, Callable

# Assumes that `categorized_functions` and `categorized_functions_with_details`
# are defined and populated as in the previous response.

categories = [
    category
    for category in categorized_functions.keys()
    if category in categorized_functions_with_details
]
from icecream import ic

ic(categories)


def get_random_subset_from_distribution(
    distribution: Dict[str, float],  # Category name -> proportion (0.0 to 1.0)
    total_count: int,  # Total number of functions to select
    functions_source: Dict[
        str, List[Callable]
    ] = categorized_functions,  # Default to using global categorized_functions, can be changed.
    allow_duplicates: bool = False,  # If true, functions can appear more than once in the returned set.
    require_all_categories_present: bool = True,  # Require all keys in distribution to exist as categories?
) -> List[Callable]:
    """
    Selects a random subset of functions based on a specified distribution across categories.

    Args:
        distribution: A dictionary mapping category names to their desired proportion in the subset.
            Values should be floats between 0.0 and 1.0, and the sum of all proportions should ideally be 1.0.
            Keys should match the keys of the `functions_source` dictionary (e.g., "Communication", "FileManagement").
        total_count: The total number of functions to select in the subset.  If it can't be fulfilled exactly
          due to rounding, the returned list can be up to len(distribution) off.
        functions_source: A dictionary mapping category names to lists of function references (like `categorized_functions`).
            Defaults to the globally defined `categorized_functions`, but can be overridden for testing or to use a different data source.
        allow_duplicates: Whether or not the same function can be included multiple times in the result.
            If False (default), each function from the source will appear at most once.
        require_all_categories_present: If True (default), raise a ValueError if the distribution specifies categories
            not found in the `functions_source`. If False, only present categories in `functions_source` will be used.

    Returns:
        A list of function references, representing the randomly selected subset.  The length of the list
        should ideally be close to `total_count` but can be slightly off due to rounding.

    Raises:
        ValueError:
            - If `distribution` contains categories not present in `functions_source` and `require_all_categories_present` is True.
            - If the proportions in `distribution` don't sum close to 1.0.
            - If a category specified in `distribution` is empty in `functions_source`.
            - If the value for a given distribution key is < 0 or > 1.

    """

    if require_all_categories_present:
        invalid_categories = [
            cat for cat in distribution if cat not in functions_source
        ]
        if invalid_categories:
            raise ValueError(
                f"Categories specified in distribution not found in functions_source: {invalid_categories}"
            )

    # Filter out non-existent categories
    valid_categories = [cat for cat in distribution if cat in functions_source]

    if not valid_categories:
        raise ValueError("No valid categories to sample from in the distribution.")

    # Handle the case of proportions not summing to 1.0 by normalizing
    total_proportion = sum(distribution[cat] for cat in valid_categories)
    if not (
        0.99 <= total_proportion <= 1.01
    ):  # Allow slight floating-point imprecision
        print(
            "Normalizing distribution because total proportion is not approximately 1.0"
        )  # warning, but proceeds
        distribution = {
            cat: distribution[cat] / total_proportion for cat in valid_categories
        }

    counts_per_category = {}
    remaining_count = total_count

    # Determine how many items to select from each category
    for i, category in enumerate(valid_categories):
        if distribution[category] < 0 or distribution[category] > 1:
            raise ValueError(f"Invalid category weight: {distribution[category]}")

        if len(functions_source[category]) == 0:
            raise ValueError(f"Source for {category} is empty, can't sample.")

        if i == (len(valid_categories) - 1):
            n_items = remaining_count  # Last key takes up the rest
        else:
            n_items = round(distribution[category] * total_count)
            remaining_count -= n_items  # Keep track of how many remaining

        counts_per_category[category] = n_items

    # Select the functions
    selected_functions = []

    for category, count in counts_per_category.items():
        source_functions = functions_source[category]
        num_source_functions = len(source_functions)

        if allow_duplicates:
            # Use random.choices to allow duplicates (sample with replacement)
            selected_functions.extend(random.choices(source_functions, k=count))
        else:
            # Sample without replacement (unique items)
            try:
                selected_functions.extend(
                    random.sample(source_functions, min(count, num_source_functions))
                )  # handles the edge case where source_functions has less then count.
            except ValueError as e:
                print(
                    f"Warning: Can't retrieve unique samples from {category}, decreasing number of elements to retrieve. {e}"
                )  # warning, but proceeds

                selected_functions.extend(
                    random.sample(source_functions, num_source_functions)
                )  # just take everything.

    return selected_functions


# List all available categories (to show the users the keys for generating a distribution)
available_categories = list(categorized_functions.keys())

if __name__ == "__main__":
    # Example Usage

    # Ensure categorized_functions is populated
    if not categorized_functions:
        print(
            "Error: categorized_functions is not populated. Please run the initial script first."
        )
    else:
        # Example 1: 50% Communication, 50% FileManagement, 10 functions total
        distribution1 = {"Communication": 0.5, "FileManagement": 0.5}
        subset1 = get_random_subset_from_distribution(distribution1, 10)
        print("\nRandom Subset 1:")
        for func in subset1:
            print(f"  - {func.__name__} ({func.computed_tags})")
