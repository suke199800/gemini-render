# 필요한 라이브러리를 가져옵니다.
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import os
import logging

# Flask 앱 인스턴스를 생성합니다.
app = Flask(__name__)

# 기본 로깅 설정
logging.basicConfig(level=logging.INFO)

# --- 원하는 시스템 프롬프트 정의 ---
# 여기에 AI에게 부여하고 싶은 역할, 말투, 규칙 등을 자세히 넣어주세요.
SYSTEM_PROMPT = """당신은 게임 '모바일 마비노기'의 'END' 길드를 위한 친절하고 유능한 AI 비서입니다. 
길드원들의 질문에 답변하는 것이 주 역할입니다. 
항상 존댓말을 사용하고, 긍정적이고 상냥한 말투를 유지해주세요. 
그리고 사용자는 무조건 용사님이라고 부르시고
길드와 관련된 정보나 게임 내 정보에 대해 답변할 수 있습니다. , 추측성 정보 추측성정보라고 말하고 제공하세요."""
# ---------------------------------

# Gemini API 설정
model = None
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("환경 변수에서 GEMINI_API_KEY를 찾을 수 없습니다.")

    genai.configure(api_key=api_key)
    # 최신 모델 사용 권장 (gemini-1.5-flash-latest 등)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    app.logger.info("Gemini 모델이 성공적으로 로드되었습니다.")

except Exception as e:
    app.logger.error(f"Gemini API 설정 중 오류 발생: {e}")


@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        app.logger.error(f"index.html 렌더링 중 오류: {e}")
        return "HTML 페이지를 로드하는 중 오류가 발생했습니다.", 500


@app.route('/ask', methods=['POST'])
def ask_gemini():
    global model

    if not model:
        app.logger.error("API 요청 수신 실패: Gemini 모델이 로드되지 않음")
        return jsonify({"error": "Gemini 모델을 초기화하지 못했습니다. 서버 로그를 확인하세요."}), 500

    data = request.get_json()
    if not data or 'history' not in data:
        app.logger.warning(f"잘못된 요청 수신: 'history' 키가 없음. 받은 데이터: {data}")
        return jsonify({"error": "잘못된 요청 형식입니다. 'history' 키가 필요합니다."}), 400

    conversation_history = data.get('history')

    if not conversation_history or not isinstance(conversation_history, list):
        app.logger.warning(f"잘못된 요청 수신: 'history'가 비어있거나 리스트가 아님: {conversation_history}")
        return jsonify({"error": "'history'는 비어 있지 않은 리스트여야 합니다."}), 400

    # 마지막 메시지가 사용자 메시지인지 확인 (선택적이지만 권장)
    if not conversation_history[-1].get('role') == 'user':
         app.logger.warning(f"잘못된 요청 수신: history의 마지막 메시지가 user가 아님: {conversation_history[-1]}")
         return jsonify({"error": "잘못된 요청 형식입니다. 마지막 메시지는 사용자 질문이어야 합니다."}), 400

    # Gemini API 형식으로 변환 ('assistant' -> 'model')
    gemini_formatted_user_history = []
    for message in conversation_history:
        role = message.get('role')
        content = message.get('content')
        if not role or content is None:
            app.logger.warning(f"잘못된 메시지 형식 발견 (role 또는 content 누락): {message}")
            continue # 문제가 있는 메시지는 건너뛰기

        gemini_role = 'model' if role == 'assistant' else role
        gemini_formatted_user_history.append({'role': gemini_role, 'parts': [content]})

    if not gemini_formatted_user_history:
         app.logger.warning(f"처리 후 사용자 history가 비어있음. 원본: {conversation_history}")
         return jsonify({"error": "처리할 유효한 대화 내용이 없습니다."}), 400

    # --- 시스템 프롬프트를 대화 기록 앞에 추가 ---
    # Gemini는 보통 user 역할로 시스템 메시지를 시작하고 model이 답하는 형식을 선호합니다.
    initial_context = [
        {'role': 'user', 'parts': [SYSTEM_PROMPT]},
        {'role': 'model', 'parts': ["네, 알겠습니다. END 길드 비서로서 무엇을 도와드릴까요?"]} # AI가 프롬프트를 인지했다는 간단한 응답
    ]
    full_history_for_api = initial_context + gemini_formatted_user_history
    # -----------------------------------------

    app.logger.info(f"Gemini에게 전달할 총 history 개수 (프롬프트 포함): {len(full_history_for_api)}")
    # app.logger.debug(f"전달할 전체 history: {full_history_for_api}") # 디버깅 시 주석 해제

    try:
        # 수정: 전체 히스토리 전달
        response = model.generate_content(full_history_for_api)

        answer_text = ""
        # 응답 텍스트 추출 및 안전 설정 확인 로직은 동일
        if hasattr(response, 'parts') and response.parts:
             answer_text = "".join(part.text for part in response.parts)
        elif hasattr(response, 'text'):
             answer_text = response.text

        if not answer_text:
            app.logger.warning(f"Gemini로부터 비어있거나 예상치 못한 응답 수신: {response}")
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                reason = response.prompt_feedback.block_reason
                app.logger.warning(f"Gemini 응답 블록됨. 이유: {reason}")
                answer_text = f"죄송합니다. 요청하신 내용에 답변할 수 없습니다. (사유: {reason})"
            else:
                answer_text = "죄송합니다. 답변을 생성하는 데 문제가 발생했습니다."

        return jsonify({"answer": answer_text})

    except Exception as e:
        app.logger.error(f"Gemini API 호출 중 오류: {e}", exc_info=True)
        return jsonify({"error": f"Gemini API와 통신 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
