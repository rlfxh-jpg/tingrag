import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.append(".")
sys.path.append("tinyrag/llm")

from gguf_llm import GGUFLLM


class ChatHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)
        payload = json.loads(body.decode("utf-8"))

        assert payload["model"] == "qwythos-local"
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"

        response = {
            "choices": [
                {
                    "message": {
                        "content": "北京是中国的首都。"
                    }
                }
            ]
        }
        data = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        return


def test_http_gguf_llm():
    server = HTTPServer(("127.0.0.1", 0), ChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        llm = GGUFLLM(
            model_id_key="qwythos-local",
            base_url=base_url,
            api_key="dummy",
            timeout=10,
        )
        response = llm.generate("请介绍一下北京")
        assert response == "北京是中国的首都。"
        print(response)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    test_http_gguf_llm()
