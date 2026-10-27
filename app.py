import streamlit as st
import tempfile
import os
from logistics_agent import (
    create_order_direct,
    load_orders,
    get_audio_response_path,
    is_valid_order_text,
)
 
st.set_page_config(page_title="LogiFlow — Логистический ассистент", layout="wide")
 
if 'last_audio_path' not in st.session_state:
    st.session_state.last_audio_path = None
if 'consultant_messages' not in st.session_state:
    st.session_state.consultant_messages = []
 
# Блок 1: Приём заявок
with st.container():
    st.title("Приём заявок на перевозку")
    mode = st.radio("Выберите способ ввода:", ["Текстовый ввод", "Голосовое сообщение"])
 
    if mode == "Текстовый ввод":
        user_input = st.text_area(
            "Опишите заявку:",
            "Забрать 500 кг яблок из Твери, доставить в Москву к пятнице",
            height=100,
        )
        if st.button("Создать заявку", type="primary", key="disp_text"):
            # Проверка через GigaChat
            with st.spinner("Проверяю заявку..."):
                is_valid, message = is_valid_order_text(user_input)
                if not is_valid:
                    st.warning(f"Заявка не зарегистрирована. {message}")
                else:
                    with st.spinner("Создаю заявку..."):
                        response = create_order_direct(user_input)
                        st.success(response)
                        
                        try:
                            from logistics_agent import text_to_audio
                            audio_path = text_to_audio.invoke(response[:300])
                            if audio_path and os.path.exists(str(audio_path)):
                                st.session_state.last_audio_path = str(audio_path)
                        except Exception:
                            pass
 
    else:
        uploaded_file = st.file_uploader("Загрузите аудиофайл (.wav)", type=["wav"])
        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")
            if st.button("Создать заявку", type="primary", key="disp_audio"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                with st.spinner("Распознаю речь..."):
                    from logistics_agent import audio_to_text, create_order_direct, text_to_audio
                    
                    # Распознаём аудио в текст
                    recognized = audio_to_text.invoke(f"Распознай: {tmp_path}")
                    
                    # Проверка на ошибку распознавания
                    if not recognized or "Ошибка" in recognized or "Не удалось" in recognized:
                        st.warning("Заявка не зарегистрирована. Не удалось распознать речь. Убедитесь, что аудиофайл содержит чёткую заявку.")
                    else:
                        st.info(f"Распознанный текст: {recognized}")
                        
                        # Проверка через GigaChat (та же функция, что и для текста)
                        with st.spinner("Проверяю заявку..."):
                            is_valid, message = is_valid_order_text(recognized)
                            if not is_valid:
                                st.warning(f"Заявка не зарегистрирована. {message}")
                            else:
                                with st.spinner("Создаю заявку..."):
                                    response = create_order_direct(recognized)
                                    st.success(response)
                                    
                                    try:
                                        audio_path = text_to_audio.invoke(response[:300])
                                        if audio_path and os.path.exists(str(audio_path)):
                                            st.session_state.last_audio_path = str(audio_path)
                                    except Exception as e:
                                        print(f"Audio synthesis error: {e}")
 
                os.unlink(tmp_path)
    
    if st.session_state.last_audio_path and os.path.exists(st.session_state.last_audio_path):
        st.write("Голосовой ответ ассистента:")
        st.audio(st.session_state.last_audio_path, format="audio/wav")
 
st.divider()
 
# Блок 2: Таблица заявок
with st.container():
    st.title("Таблица заявок")
 
    df = load_orders()
    if df.empty:
        st.info("Заявок пока нет. Создайте первую заявку выше.")
    else:
        st.dataframe(df, use_container_width=True)
        csv_data = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Скачать CSV", data=csv_data, file_name="orders.csv", mime="text/csv")
 
st.divider()
 
# Блок 3: Консультация
with st.container():
    st.title("Консультация по логистике")
    st.write("Задайте вопрос о правилах перевозки, тарифах и документах")
 
    if len(st.session_state.consultant_messages) != 0:
        if st.button("Очистить чат"):
            st.session_state.consultant_messages = []
            st.rerun()
 
        for msg in st.session_state.consultant_messages:
            if msg["role"] == "user":
                st.write(f"**Вы**: {msg['content']}")
            else:
                st.write(f"**Аналитик**: {msg['content']}")
 
    question = st.text_input(
        "Ваш вопрос:",
        "Какие документы нужны для перевозки опасных грузов?",
    )
    if st.button("Получить ответ", type="primary", key="analyst"):
        if question.strip():
            st.session_state.consultant_messages.append({"role": "user", "content": question})
            with st.spinner("Ищу ответ в базе знаний..."):
                from logistics_agent import analyst_executor
                response = analyst_executor.invoke({"input": question})
                answer = response["output"]
                st.session_state.consultant_messages.append({"role": "assistant", "content": answer})
                st.rerun()