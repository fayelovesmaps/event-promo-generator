"""
Event Promo Generator — Backend Server
Proxies API calls to:
  - DashScope Qwen Image 2.0 Pro (poster/image generation)
  - DashScope HappyHorse T2V (video generation, async with polling)

Run:  python3 server.py
Then open http://localhost:8765

Requires:
  - DASHSCOPE_API_KEY env var (or enter in UI)
"""

import json
import os
import sys
import time
import base64
import urllib.request
import urllib.error
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

PORT = int(os.environ.get("PORT", 8765))
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── API Endpoints ───

DASHSCOPE_ENDPOINTS = {
    "mainland": "https://dashscope.aliyuncs.com",
    "intl": "https://dashscope-intl.aliyuncs.com"
}


class Handler(SimpleHTTPRequestHandler):
    """Serves static files and handles /api/* routes."""

    def do_POST(self):
        if self.path == "/api/generate-image":
            self._handle_generate_image()
        elif self.path == "/api/generate-video":
            self._handle_generate_video()
        elif self.path == "/api/poll-video":
            self._handle_poll_video()
        elif self.path == "/api/defaults":
            self._handle_defaults()
        elif self.path == "/api/suggest-styles":
            self._handle_suggest_styles()
        else:
            self._json_response(404, {"error": "Not found"})

    def do_GET(self):
        if self.path == "/api/defaults":
            self._handle_defaults()
        elif self.path.startswith("/outputs/"):
            super().do_GET()
        elif self.path == "/" or self.path == "/index.html":
            self.path = "/index.html"
            super().do_GET()
        else:
            super().do_GET()

    # ─── Defaults ───

    def _handle_defaults(self):
        self._json_response(200, {
            "hasDashscopeKey": bool(os.environ.get("DASHSCOPE_API_KEY")),
        })

    # ─── Suggest Creative Directions (DashScope Qwen LLM) ───

    def _handle_suggest_styles(self):
        body = self._read_body()
        if not body:
            return

        event_name = body.get("eventName", "")
        date = body.get("date", "")
        venue = body.get("venue", "")
        performers = body.get("performers", "")
        tagline = body.get("tagline", "")
        cta = body.get("cta", "")
        region = body.get("region", "mainland")
        api_key = body.get("apiKey", "") or os.environ.get("DASHSCOPE_API_KEY", "")

        if not event_name:
            self._json_response(400, {"error": "Event name is required"})
            return
        if not api_key:
            self._json_response(400, {"error": "DashScope API key is required"})
            return

        base_url = DASHSCOPE_ENDPOINTS.get(region, DASHSCOPE_ENDPOINTS["mainland"])
        url = f"{base_url}/compatible-mode/v1/chat/completions"

        # Build event context
        event_desc = f"Event: {event_name}"
        if date: event_desc += f"\nDate & Time: {date}"
        if venue: event_desc += f"\nVenue & City: {venue}"
        if performers: event_desc += f"\nPerformers / Speakers: {performers}"
        if tagline: event_desc += f"\nTagline / Theme: {tagline}"
        if cta: event_desc += f"\nTicket / CTA: {cta}"

        system_prompt = (
            "You are a professional graphic designer specializing in event promotional posters. "
            "Given event details, propose exactly 3 creative directions for a poster. "
            "Each direction must be distinct, tailored to this specific event's nature, audience, and venue.\n\n"
            "Return ONLY valid JSON with no markdown fences. Format:\n"
            '{"directions": [\n'
            '  {\n'
            '    "name": "Short Creative Name (3-5 words)",\n'
            '    "description": "1-2 sentence visual concept tailored to this event",\n'
            '    "palette": "4-5 specific color names that suit the event mood",\n'
            '    "mood": "2-3 mood keywords",\n'
            '    "composition": "Detailed poster layout/composition description with specific elements related to the event theme",\n'
            '    "lighting": "Lighting direction, color temperature, and atmospheric effects",\n'
            '    "texture": "Material, surface finish, and tactile quality description",\n'
            '    "fontMix": "Specific font style pairings for title and body text",\n'
            '    "negation": "What the poster should NOT look like — steer away from defaults"\n'
            '  }\n'
            ']}\n\n'
            "Make each direction genuinely different in aesthetic, palette, and layout approach. "
            "Consider the event's nature, likely audience, venue character, and theme."
        )

        payload = json.dumps({
            "model": "qwen-plus",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": event_desc}
            ],
            "temperature": 0.8
        }).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())

            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            # Strip markdown fences if present
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            directions = json.loads(content)
            self._json_response(200, {"success": True, "directions": directions.get("directions", [])})

        except urllib.error.HTTPError as e:
            err_body = e.read().decode() if e.readable() else ""
            try:
                err_json = json.loads(err_body)
                msg = err_json.get("message", err_body[:500])
            except Exception:
                msg = err_body[:500]
            self._json_response(e.code, {"error": f"DashScope LLM Error ({e.code}): {msg}"})
        except json.JSONDecodeError as e:
            self._json_response(500, {"error": f"Failed to parse LLM response: {str(e)}"})
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    # ─── Image Generation (DashScope Qwen Image 2.0 Pro) ───

    def _handle_generate_image(self):
        body = self._read_body()
        if not body:
            return

        prompt = body.get("prompt", "")
        negative_prompt = body.get("negativePrompt", "") or body.get("negative_prompt", "")
        size = body.get("size", "1024*1536")
        n = body.get("n", 1)
        region = body.get("region", "mainland")
        api_key = body.get("apiKey", "") or os.environ.get("DASHSCOPE_API_KEY", "")

        if not prompt:
            self._json_response(400, {"error": "prompt is required"})
            return
        if not api_key:
            self._json_response(400, {"error": "DashScope API key is required"})
            return

        base_url = DASHSCOPE_ENDPOINTS.get(region, DASHSCOPE_ENDPOINTS["mainland"])
        url = f"{base_url}/api/v1/services/aigc/multimodal-generation/generation"

        # Convert size format: "1024x1536" -> "1024*1536"
        size_param = size.replace("x", "*")

        parameters = {
            "size": size_param,
            "n": n
        }
        if negative_prompt:
            parameters["negative_prompt"] = negative_prompt

        payload = json.dumps({
            "model": "qwen-image-2.0-pro",
            "input": {
                "messages": [
                    {"role": "user", "content": [{"text": prompt}]}
                ]
            },
            "parameters": parameters
        }).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        )

        try:
            # Synchronous call — qwen-image-2.0-pro does not support async
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read())

            # Extract image URL from synchronous response
            output = result.get("output", {})
            # Try multimodal-generation response format: output.choices[].message.content[].image
            choices = output.get("choices", [])
            image_urls = []
            for choice in choices:
                msg = choice.get("message", {})
                for item in msg.get("content", []):
                    if "image" in item:
                        image_urls.append(item["image"])
            # Fallback: try results[] format
            if not image_urls:
                for r in output.get("results", []):
                    if r.get("url"):
                        image_urls.append(r["url"])
            # Fallback: result_url
            if not image_urls and output.get("result_url"):
                image_urls.append(output["result_url"])

            if not image_urls:
                self._json_response(500, {"error": "No image URL in response", "raw": result})
                return

            images = []
            for i, img_url in enumerate(image_urls):
                images.append({
                    "remoteUrl": img_url,
                    "filename": f"poster-{int(time.time())}-{i}.png"
                })

            self._json_response(200, {"success": True, "images": images})

        except urllib.error.HTTPError as e:
            err_body = e.read().decode() if e.readable() else ""
            try:
                err_json = json.loads(err_body)
                msg = err_json.get("message", err_body[:500])
            except:
                msg = err_body[:500]
            self._json_response(e.code, {"error": f"DashScope Image API Error ({e.code}): {msg}"})
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _poll_image_task(self, api_key, region, task_id, max_wait=180):
        """Poll a DashScope image task until done. Returns list of URLs or error string."""
        base_url = DASHSCOPE_ENDPOINTS.get(region, DASHSCOPE_ENDPOINTS["mainland"])
        url = f"{base_url}/api/v1/tasks/{task_id}"
        start = time.time()

        while time.time() - start < max_wait:
            time.sleep(3)
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    result = json.loads(resp.read())
                output = result.get("output", {})
                status = output.get("task_status", "")

                if status == "SUCCEEDED":
                    # Extract image URLs from results
                    results = output.get("results", [])
                    urls = [r.get("url") or r.get("b64_image", "") for r in results if r.get("url")]
                    if not urls:
                        # Try alternate response format
                        urls = [output.get("result_url", "")]
                        urls = [u for u in urls if u]
                    return urls if urls else ["No image URL in response"]

                elif status in ("FAILED", "CANCELED"):
                    return output.get("message", f"Task {status}")

            except Exception:
                pass  # Retry on transient errors

        return "Image generation timed out after 3 minutes"

    # ─── Video Generation (DashScope HappyHorse T2V) ───

    def _handle_generate_video(self):
        body = self._read_body()
        if not body:
            return

        prompt = body.get("prompt", "")
        duration = body.get("duration", 5)
        resolution = body.get("resolution", "720P")
        ratio = body.get("ratio", "9:16")  # Default vertical for speaker videos
        region = body.get("region", "mainland")
        api_key = body.get("apiKey", "") or os.environ.get("DASHSCOPE_API_KEY", "")
        reference_image = body.get("referenceImage", None)  # Speaker reference photo URL

        if not prompt:
            self._json_response(400, {"error": "prompt is required"})
            return
        if not api_key:
            self._json_response(400, {"error": "DashScope API key is required"})
            return

        base_url = DASHSCOPE_ENDPOINTS.get(region, DASHSCOPE_ENDPOINTS["mainland"])
        url = f"{base_url}/api/v1/services/aigc/video-generation/video-synthesis"

        # Build request body — HappyHorse 1.1 with native audio
        parameters = {
            "resolution": resolution,
            "duration": int(duration),
            "ratio": ratio
        }

        # If a reference image is provided, use R2V model for character identity preservation
        if reference_image:
            model = "happyhorse-1.1-r2v"
            input_data = {
                "prompt": prompt,
                "media": [
                    {"type": "reference_image", "url": reference_image}
                ]
            }
        else:
            model = "happyhorse-1.1-t2v"
            input_data = {
                "prompt": prompt
            }

        payload = json.dumps({
            "model": model,
            "input": input_data,
            "parameters": parameters
        }).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable"
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())

            task_id = result.get("output", {}).get("task_id")
            if not task_id:
                self._json_response(500, {"error": "No task_id returned", "raw": result})
                return

            self._json_response(200, {
                "success": True,
                "taskId": task_id,
                "status": result.get("output", {}).get("task_status", "PENDING"),
                "region": region
            })

        except urllib.error.HTTPError as e:
            err_body = e.read().decode() if e.readable() else ""
            try:
                err_json = json.loads(err_body)
                msg = err_json.get("message", err_body[:500])
            except:
                msg = err_body[:500]
            self._json_response(e.code, {"error": f"DashScope API Error ({e.code}): {msg}"})
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    # ─── Poll Video Task ───

    def _handle_poll_video(self):
        body = self._read_body()
        if not body:
            return

        task_id = body.get("taskId", "")
        region = body.get("region", "mainland")
        api_key = body.get("apiKey", "") or os.environ.get("DASHSCOPE_API_KEY", "")

        if not task_id:
            self._json_response(400, {"error": "taskId is required"})
            return
        if not api_key:
            self._json_response(400, {"error": "DashScope API key is required"})
            return

        base_url = DASHSCOPE_ENDPOINTS.get(region, DASHSCOPE_ENDPOINTS["mainland"])
        url = f"{base_url}/api/v1/tasks/{task_id}"

        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {api_key}"
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read())

            output = result.get("output", {})
            status = output.get("task_status", "UNKNOWN")
            video_url = output.get("video_url", "")

            response_data = {
                "success": True,
                "status": status,
                "taskId": task_id
            }

            if video_url:
                response_data["videoUrl"] = video_url
            if output.get("message"):
                response_data["message"] = output["message"]
            if result.get("usage"):
                response_data["usage"] = result["usage"]

            self._json_response(200, response_data)

        except urllib.error.HTTPError as e:
            err_body = e.read().decode() if e.readable() else ""
            self._json_response(e.code, {"error": f"Poll failed ({e.code}): {err_body[:300]}"})
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    # ─── Helpers ───

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            return json.loads(raw)
        except Exception as e:
            self._json_response(400, {"error": f"Invalid JSON body: {e}"})
            return None

    def _json_response(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {format % args}\n")


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def main():
    os.chdir(Path(__file__).parent)
    server = ReusableHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"\n  Event Promo Generator")
    print(f"  ─────────────────────────────")
    print(f"  Server: http://localhost:{PORT}")
    print(f"  Outputs: {OUTPUT_DIR.resolve()}")
    print(f"  DashScope key from env: {'Yes' if os.environ.get('DASHSCOPE_API_KEY') else 'No (enter in UI)'}")
    print(f"  Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
