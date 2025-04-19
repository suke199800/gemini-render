# 필요한 라이브러리를 가져옵니다.
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import os
# API 키를 환경 변수에서 직접 로드하므로 python-dotenv는 서버 환경에서 필수는 아닙니다.
# 하지만 로컬 개발 시에는 유용할 수 있습니다.

# Flask 앱 인스턴스를 생성합니다.
app = Flask(__name__)

# Gemini API 설정
try:
    # Render와 같은 배포 환경에서는 환경 변수에서 직접 API 키를 읽어옵니다.
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("환경 변수에서 GEMINI_API_KEY를 찾을 수 없습니다.")

    genai.configure(api_key=api_key)

    # 사용할 Gemini 모델 설정
    model = genai.GenerativeModel('gemini-2.5-pro-preview-03-25')
    print("Gemini 모델이 성공적으로 로드되었습니다.") # 서버 로그에 성공 메시지 출력

except Exception as e:
    print(f"Gemini API 설정 중 오류 발생: {e}")
    # 실제 운영 환경에서는 오류 로깅 및 처리를 더 견고하게 해야 합니다.
    model = None # 모델 로딩 실패 시 None으로 설정

# 루트 URL ('/') 접속 시 HTML 페이지를 보여줍니다.
@app.route('/')
def index():
    # 'templates' 폴더 안의 'index.html' 파일을 렌더링합니다.
    # templates 폴더가 app.py와 같은 레벨에 있어야 합니다.
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"index.html 렌더링 중 오류: {e}")
        return "HTML 페이지를 로드하는 중 오류가 발생했습니다.", 500


# '/ask' URL로 POST 요청이 오면 Gemini API를 호출합니다.
@app.route('/ask', methods=['POST'])
def ask_gemini():
    # 모델이 로드되지 않았으면 오류 반환
    if not model:
        return jsonify({"error": "Gemini 모델을 초기화하지 못했습니다. 서버 로그를 확인하세요."}), 500

    # 클라이언트(HTML 페이지)로부터 JSON 데이터에서 'question'을 가져옵니다.
    data = request.get_json()
    if not data or 'question' not in data:
         return jsonify({"error": "잘못된 요청 형식입니다. 'question' 키가 필요합니다."}), 400

    question = data.get('question')

    # 질문 내용이 비어있는지 확인
    if not question:
        return jsonify({"error": "질문 내용이 없습니다."}), 400 # 잘못된 요청 응답

    try:
        # Gemini 모델에 질문을 보내고 응답을 생성합니다.
        # stream=True 를 사용하면 더 긴 응답을 처리하거나 실시간 표시가 가능하지만,
        # 현재 프론트엔드에서는 전체 응답을 한 번에 받으므로 stream=False (기본값) 사용
        response = model.generate_content(question)

        # response.text 대신 response.parts 확인 (더 안정적)
        answer_text = "".join(part.text for part in response.parts) if hasattr(response, 'parts') else response.text

        # 응답 텍스트를 JSON 형태로 클라이언트에 돌려줍니다.
        return jsonify({"answer": answer_text})

    except Exception as e:
        # API 호출 중 오류 발생 시 오류 메시지를 돌려줍니다.
        print(f"Gemini API 호출 중 오류: {e}")
        # 사용자에게는 조금 더 일반적인 오류 메시지를 보여주는 것이 좋을 수 있습니다.
        return jsonify({"error": f"Gemini API와 통신 중 오류가 발생했습니다."}), 500

# 이 부분은 로컬 개발 시 `python app.py`로 직접 실행할 때 사용됩니다.
# Render와 같은 WSGI 서버 환경(Gunicorn)에서는 이 부분이 직접 호출되지 않습니다.
# Gunicorn이 Flask 앱 객체(`app`)를 직접 찾아 실행합니다.
if __name__ == '__main__':
    # debug=True는 개발 중에만 사용하고, 실제 배포 시에는 Gunicorn이 관리합니다.
    # host='0.0.0.0'은 로컬 네트워크의 다른 기기에서도 접속 가능하게 합니다.
    # port=5000은 Flask 기본 포트입니다. Render는 보통 다른 포트를 사용합니다.
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    # Render는 PORT 환경 변수를 제공하므로 위와 같이 사용하는 것이 좋습니다.
    # debug=False 로 설정하는 것이 운영 환경에 더 가깝습니다.
