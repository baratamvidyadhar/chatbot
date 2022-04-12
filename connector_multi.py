from fbchat import log
from fbchat import Client
from fbchat.models import *
from credentials import getusername
from credentials import getPassword
from response_file import chatbot_response


class EchoBot(Client):

    def onMessage(self, author_id=None, message_object=None, thread_id=None, thread_type=ThreadType.USER, **kwargs):
        self.markAsRead(author_id)
        log.info("Message {} from {} in {}".format(message_object, thread_id, thread_type))
        msgText = message_object.text
        msgText = str(msgText).lower()
        if msgText == "byee":
            reply = "bye!"
            self.send(Message(text=reply), thread_id=thread_id, thread_type=thread_type)
        elif msgText == "good night" or msgText == "night!":
            reply = "good night!sweet dreams bye"
            self.send(Message(text=reply), thread_id=thread_id, thread_type=thread_type)
        elif author_id != self.uid:
            reply = chatbot_response(msgText)
            self.send(Message(text=reply), thread_id=thread_id, thread_type=thread_type)
        self.markAsDelivered(author_id, thread_id)


username = getusername()
password = getPassword()
client = EchoBot(username, password)
print(client.listen())