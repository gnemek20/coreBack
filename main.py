from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from collections import defaultdict, Counter
from itertools import combinations, permutations

import cv2
import numpy as np
import os
import math

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/help-core/")
async def help(files: list[UploadFile] = File(...), job: str = Form(...), target_skills: list[str] = Form(...)):
  try:
    displays = []
    for file in files:
      contents = await file.read()
      npimg = np.frombuffer(contents, np.uint8)
      display = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
      displays.append(display)

    base_path = "static"
    table_template = cv2.imread(os.path.join(base_path, "templates", "table.png"))
    empty_template = cv2.imread(os.path.join(base_path, "templates", "empty.png"))
    job_dir = os.path.join(base_path, "jobs", job)

    results = []
    for display in displays:
      table = get_table(display, table_template)
      cores = get_cores(table, empty_template)
      cores = get_enhanced_cores(cores)
      core_skills = get_skills(cores)
      parsed_skills = parse_skills(core_skills, job_dir)

      if len(parsed_skills) < 1:
        continue

      _, skill_names = zip(*parsed_skills)
      combinations = find_combinations(skill_names, target_skills)
      print(combinations)
      results.append(combinations)

    return JSONResponse(content={
      "success": True,
      "message": "코어 조합 분석 완료",
      "results": results
    })

  except Exception as e:
    return JSONResponse(status_code=500, content={
      "success": False,
      "message": f"분석 중 오류 발생: {str(e)}"
    })

def get_table(display, template):
  gray_display = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
  gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

  # 테이블 템플릿 매칭
  result = cv2.matchTemplate(gray_display, gray_template, cv2.TM_CCOEFF_NORMED)
  _, _, _, max_loc = cv2.minMaxLoc(result)

  # 매칭 영역 좌표
  top_left = max_loc
  h, w = gray_template.shape
  bottom_right = (top_left[0] + w, top_left[1] + h)

  # 매칭 영역 추출
  cropped = display[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
  return cropped

def get_cores(table, template):
  cores = []

  gray_table = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)
  gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

  # 윤곽선 검출
  blurred = cv2.GaussianBlur(table, (5, 5), 0)
  edges = cv2.Canny(blurred, threshold1=90, threshold2=180)
  contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  for cnt in contours:
    # 소형 객체 필터링
    area = cv2.contourArea(cnt)
    if area < 500:
      continue

    # 윤곽선 근사화
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # 5 ~ 9각형 체크
    if 5 <= len(approx) <= 9:
      x, y, w, h = cv2.boundingRect(approx)
      aspect_ratio = w / h
      # 가로/세로 비율 체크
      if 0.7 < aspect_ratio < 1.3:
        # 빈 코어 템플릿 매칭
        cropped = gray_table[y:y+h, x:x+w]
        h, w = cropped.shape
        gray_template = cv2.resize(gray_template, (w, h))
        result = cv2.matchTemplate(cropped, gray_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        # 코어 추출
        if max_val < 0.5:
          cores.append(table[y:y+h, x:x+w])
  return cores

def get_enhanced_cores(cores):
  enhanced_cores = []

  dark_color = np.array([0, 0, 0])
  light_color = np.array([94, 115, 113])

  for c in cores:
    # 강화 코어 색 마스킹
    mask = cv2.inRange(c, dark_color, light_color)

    # 코어 배경 좌표
    h, w = mask.shape
    x = w // 6
    y = (h // 2) - 5

    # 코어 배경 분석
    white = 0
    black = 0
    for i in range(10):
      if mask[y+i, x] == 255:
        white += 1
      else:
        black += 1

    # 마스킹 체크
    if white > black:
      enhanced_cores.append(c)
  return enhanced_cores

def get_skills(cores):
  skills = []
  for idx, c in enumerate(cores):
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    color = np.array([0, 0, 0])

    # 스킬 테두리 마스킹
    mask = cv2.inRange(hsv, color, color)

    # 윤곽선 검출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 무효 객체 필터링
    if not contours:
      continue

    # 대형 객체 좌표
    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)

    # 스킬 추출
    cropped = c[y:y+h, x:x+w]
    skills.append(cropped)

  return skills

def parse_skills(core_skills, job):
  def generate_skills(files):
    def apply_border(image):
      color = [221, 221, 204]
      white = [255, 255, 255]
      for x in range(1, 31):
        image[1, x] = color
        image[30, x] = color
        image[2, x] = white
        image[29, x] = white
      for y in range(2, 30):
        image[y, 1] = color
        image[y, 30] = color
        image[y, 2] = white
        image[y, 29] = white
      return image

    skills = []
    size = 32

    images, file_names = zip(*files)
    resized_images = []
    for image in images:
      image = cv2.resize(image, (size, size))
      resized_images.append(image)
    files = list(zip(resized_images, file_names))

    # 삼각형 마스크 생성
    triangle_mask = np.zeros((size, size), dtype=np.uint8)
    pts = np.array([[0, 0], [31, 0], [16 ,16]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(triangle_mask, [pts], 255)

    combination_files = permutations(files, 3)
    for combo_file in combination_files:
      combo_images, combo_file_names = zip(*combo_file)
      canvas = np.zeros_like(combo_images[0])

      # 중앙 이미지 배치
      for center in range(3):
        canvas[:, :, center] = np.where(triangle_mask == 255, combo_images[1][:, :, center], canvas[:, :, center])

      # 좌우 이미지 배치
      for y in range(size):
        for x in range(size):
          if triangle_mask[y, x] == 0:
            if x < 16:
              canvas[y, x] = combo_images[0][y, x]
            else:
              canvas[y, x] = combo_images[2][y, x]
      canvas = apply_border(canvas)
      skills.append((canvas, combo_file_names))
    return skills

  def find_matching_skill(core_skill, generated_skills):
    size = 32
    core_skill = cv2.resize(core_skill, (size, size))

    vals = []
    for generated_skill in generated_skills:
      generated_skill = cv2.resize(generated_skill, (size, size))

      # 스킬 템플릿 매칭
      result = cv2.matchTemplate(core_skill, generated_skill, cv2.TM_CCOEFF_NORMED)
      _, max_val, _, _ = cv2.minMaxLoc(result)

      vals.append(max_val)
    most_val_index = vals.index(max(vals))
    return most_val_index

  # 직업 스킬 수집
  files = []
  for file_name in os.listdir(job):
    if file_name.lower().endswith('.png'):
      path = os.path.join(job, file_name)
      image = cv2.imread(path)
      files.append((image, file_name[:-4]))

  # 스킬 조합 생성
  generated_skills = generate_skills(files)
  generated_skill_images, _ = zip(*generated_skills)

  # 스킬 분석
  parsed_skills = []
  for core_skill in core_skills:
    skill_index = find_matching_skill(core_skill, generated_skill_images)
    parsed_skills.append(generated_skills[skill_index])

  return parsed_skills

def find_combinations(core_skills, target_skills):
  def get_same_skill_count_avg(main_skill_indices, min_len, combo):
    same_skill_total_count = 0
    for i in range(min_len):
      main_skill = core_skills[combo[i]][0]
      same_skill_count = len(main_skill_indices[main_skill])
      same_skill_total_count += same_skill_count
    return same_skill_total_count / min_len

  n = len(target_skills)

  # 메인 스킬 인덱싱
  main_skill_indices = defaultdict(list)
  for i, core_skill in enumerate(core_skills):
    main_skill_indices[core_skill[0]].append(i)

  all_main_skills = list(main_skill_indices.keys())
  min_case_count = math.ceil(2 * n / 3)
  max_case_count = min(2 * n, len(all_main_skills)) + 1

  valid_combinations = []
  min_len = float('inf')

  # 조합 생성
  for cnt in range(min_case_count, max_case_count):
    for main_combo in combinations(all_main_skills, cnt):
      index_combinations = [[]]
      for skill in main_combo:
        new_combos = []
        for prev in index_combinations:
          for idx in main_skill_indices[skill]:
            new_combos.append(prev + [idx])
        index_combinations = new_combos

      # 조건 기반 조합 추출
      for combo in index_combinations:
        skill_counter = Counter()
        for idx in combo:
          skill_counter.update(core_skills[idx])
        if all(skill_counter[skill] >= 2 for skill in target_skills):
          if len(combo) < min_len:
            min_len = len(combo)
            same_skill_avg = get_same_skill_count_avg(main_skill_indices, min_len, combo)
            valid_combinations = [(combo, same_skill_avg)]
          elif len(combo) == min_len:
            same_skill_avg = get_same_skill_count_avg(main_skill_indices, min_len, combo)
            valid_combinations.append((combo, same_skill_avg))
          else:
            break

  final_combinations = []
  max_avg = float(0)
  for valid_combo in valid_combinations:
    combination = valid_combo[0]
    avg = valid_combo[1]
    if avg > max_avg:
      max_avg = avg
      final_combinations = [core_skills[combo] for combo in combination]
  return final_combinations