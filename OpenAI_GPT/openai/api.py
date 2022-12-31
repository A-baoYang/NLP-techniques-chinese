import configparser

from pyChatGPT import ChatGPT


class ChatGPTAPI:
    def __init__(self):
        config = configparser.ConfigParser(interpolation=None)
        config.read_file(open("secret.cfg"))
        self.session_token = config.get("OpenAI", "session_token")
        self.csrf_token = config.get("OpenAI", "csrf_token")
        self.email = config.get("OpenAI", "email")
        self.password = config.get("OpenAI", "password")
        self.login()

    def login(self, **kwargs):
        from collections import defaultdict

        kwargs = defaultdict(lambda: None, kwargs)
        self.chatgpt_api = ChatGPT(
            # session_token=self.session_token,
            # csrf_token=self.csrf_token,
            email=self.email,
            password=self.password,
            auth_type="google",
        )  # auth with session token

    def get(self, msg):
        return self.chatgpt_api.send_message(msg)["message"]

    def refresh(self):
        self.chatgpt_api.refresh_auth()
        self.login()


# ====methods====
# resp = chatgpt_api.send_message("write a song")
# print(resp["message"])

# chatgpt_api.refresh_auth()  # refresh the authorization token
# chatgpt_api.reset_conversation()  # reset the conversation
