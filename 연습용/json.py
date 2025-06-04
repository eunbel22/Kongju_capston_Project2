#변경

def process_json_data(self, data, filename: str):
    """
    크롤링 JSON 데이터 처리
    1) 과거 포맷: {"menu": ..., "content": ...}
    2) 새로운 포맷: {"category": ..., "title": ..., "content": ...}
    3) list 형태: 여러 아이템(each either 포맷1 or 포맷2)
    """
    # --- 포맷1: 단일 dict, 'menu' + 'content' 키가 있을 때 ---
    if isinstance(data, dict) and "menu" in data and "content" in data:
        menu = data["menu"]
        content = data["content"]
        cleaned = self.clean_html_content(content)
        sections = self.extract_meaningful_sections(cleaned, menu)
        for section in sections:
            if section["content"].strip():
                doc = {
                    "content": section["content"],
                    "source": filename,
                    "category": menu,
                    "section": section["title"],
                    "metadata": {
                        "file": filename,
                        "menu": menu,
                        "section": section["title"],
                        "type": "website_content"
                    }
                }
                self.documents.append(doc)

    # --- 포맷2: 단일 dict, 'category' + 'title' + 'content' 키가 있을 때 ---
    elif isinstance(data, dict) and "category" in data and "content" in data and "title" in data:
        category = data["category"]
        content = data["content"]
        title = data["title"]
        cleaned = self.clean_html_content(content)
        sections = self.extract_meaningful_sections(cleaned, category)
        for section in sections:
            if section["content"].strip():
                doc = {
                    "content": section["content"],
                    "source": filename,
                    "category": category,
                    "section": section["title"] or title,
                    "metadata": {
                        "file": filename,
                        "category": category,
                        "title": title,
                        "section": section["title"],
                        "type": "website_content"
                    }
                }
                self.documents.append(doc)

    # --- 리스트 형태: item마다 포맷1 혹은 포맷2로 재귀 처리 ---
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                # item이 포맷1인 경우
                if "menu" in item and "content" in item:
                    self.process_json_data(item, f"{filename}[{idx}]")
                # item이 포맷2인 경우
                elif "category" in item and "content" in item and "title" in item:
                    self.process_json_data(item, f"{filename}[{idx}]")
                else:
                    # 호환성 유지: legacy 방식으로 처리
                    self.process_legacy_json_data(item, f"{filename}[{idx}]")
            else:
                # dict가 아닌 경우(예: 문자열, 숫자 등)는 건너뜀
                continue

    # --- legacy 방식: 기타 dict 구조 처리 ---
    else:
        self.process_legacy_json_data(data, filename)

    def process_legacy_json_data(self, data, filename: str):
        """
        기존 JSON 형식 처리 (호환성 유지)
        - key,value 쌍을 문서로 저장
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (str, int, float)):
                    doc = {
                        "content": f"{key}: {value}",
                        "source": filename,
                        "category": key,
                        "metadata": {"file": filename, "key": key, "type": "legacy"}
                    }
                    self.documents.append(doc)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        doc = {
                            "content": f"{key} - {sub_key}: {sub_value}",
                            "source": filename,
                            "category": key,
                            "metadata": {"file": filename, "key": f"{key}.{sub_key}", "type": "legacy"}
                        }
                        self.documents.append(doc)
#수정 전
    def process_json_data(self, data, filename):
        """공주대학교 웹사이트 크롤링 JSON 데이터 처리"""
        if isinstance(data, dict) and "menu" in data and "content" in data:
            # 웹사이트 크롤링 데이터 형식 처리
            menu = data["menu"]
            content = data["content"]

            # HTML 태그 제거 및 텍스트 정리
            cleaned_content = self.clean_html_content(content)

            # 의미있는 섹션들로 분할
            sections = self.extract_meaningful_sections(cleaned_content, menu)

            for section in sections:
                if section["content"].strip():  # 빈 내용 제외
                    doc = {
                        "content": section["content"],
                        "source": filename,
                        "category": menu,
                        "section": section["title"],
                        "metadata": {
                            "file": filename,
                            "menu": menu,
                            "section": section["title"],
                            "type": "website_content"
                        }
                    }
                    self.documents.append(doc)

        elif isinstance(data, list):
            # 리스트 형태의 데이터 처리 (기존 방식 유지)
            for i, item in enumerate(data):
                if isinstance(item, dict) and "menu" in item and "content" in item:
                    self.process_json_data(item, f"{filename}[{i}]")

        else:
            # 기존 방식으로 처리 (이전 JSON 형식 호환성)
            self.process_legacy_json_data(data, filename)


    def process_legacy_json_data(self, data, filename):
        """기존 JSON 형식 처리 (호환성 유지)"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (str, int, float)):
                    doc = {
                        "content": f"{key}: {value}",
                        "source": filename,
                        "category": key,
                        "metadata": {"file": filename, "key": key, "type": "legacy"}
                    }
                    self.documents.append(doc)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        doc = {
                            "content": f"{key} - {sub_key}: {sub_value}",
                            "source": filename,
                            "category": key,
                            "metadata": {"file": filename, "key": f"{key}.{sub_key}", "type": "legacy"}
                        }
                        self.documents.append(doc)



def extract_general_sections #이부분도

#전
    sections = []

    # 문장 단위로 분할 (너무 긴 내용 방지)
    sentences = re.split(r'[.!?]\s+', content)

    # 의미있는 길이의 문장들을 그룹화
    current_section = []
    current_length = 0

#후
    sections: List[Dict] = []
    sentences = re.split(r'[.!?]\s+', content)
    current_section: List[str] = []
    current_length = 0