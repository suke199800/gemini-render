from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import os
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

model = None
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("환경 변수에서 GEMINI_API_KEY를 찾을 수 없습니다.")

    genai.configure(api_key=api_key)
    # 최신 모델 사용 권장 (예: gemini-1.5-flash-latest)
    # 사용자가 이전에 사용한 모델명으로 유지합니다. 필요시 변경하세요.
    model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
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
    global model # 전역 model 변수 사용 명시

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

    if not conversation_history[-1].get('role') == 'user':
         app.logger.warning(f"잘못된 요청 수신: history의 마지막 메시지가 user가 아님: {conversation_history[-1]}")
         return jsonify({"error": "잘못된 요청 형식입니다. 마지막 메시지는 사용자 질문이어야 합니다."}), 400

    gemini_formatted_history = []
    for message in conversation_history:
        role = message.get('role')
        content = message.get('content')
        if not role or content is None: # content가 빈 문자열일 수도 있으므로 None 체크
            app.logger.warning(f"잘못된 메시지 형식 발견 (role 또는 content 누락): {message}")
            # 문제가 있는 메시지는 건너뛰거나 오류 반환 결정 가능
            continue # 일단 건너뛰기
            # return jsonify({"error": "대화 기록에 잘못된 형식의 메시지가 포함되어 있습니다."}), 400

        gemini_role = 'model' if role == 'assistant' else role
        gemini_formatted_history.append({'role': gemini_role, 'parts': [content]})

    if not gemini_formatted_history:
         app.logger.warning(f"처리 후 Gemini history가 비어있음. 원본: {conversation_history}")
         return jsonify({"error": "처리할 유효한 대화 내용이 없습니다."}), 400

    app.logger.info(f"Gemini에게 전달할 history 개수: {len(gemini_formatted_history)}")

    try:
        response = model.generate_content(gemini_formatted_history)

        answer_text = ""
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
