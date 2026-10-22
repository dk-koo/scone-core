"""
개선된 다국어 지원 모듈
- 추가 메시지들
- 더 명확한 설명
- 에러 처리 개선
"""

LANG_KO = {
    # 기본 GUI
    'app_title': '동작 데이터 분석기 v3.0',
    'file_not_selected': '파일이 선택되지 않았습니다.',
    'select_file': '분석할 파일:',
    'open_file': '파일 열기',
    'var_select': '변수 선택',
    'select_list': '선택 목록',
    'add_to_select': '→ 목록 추가',
    'remove_from_select': '목록 삭제',
    'graph_list': 'Graph 목록',
    'add_to_graph': '→ Graph 목록 추가',
    'remove_from_graph': '목록 삭제',
    'draw_graph': 'Graph 보기',
    'draw_norm_graph': 'Normalized Graph 보기',
    'save_graph': 'Graph 저장',
    'save_norm_graph': 'Normalized Graph 저장',
    'analysis_result': '분석 결과',
    'run_analysis': '분석 실행',
    'save_result': '결과 저장',
    'data_quality': '데이터 품질',
    'statistics_options': '통계 옵션',
    
    # 성공 메시지
    'file_loaded': '파일이 성공적으로 로드되었습니다.\n왼쪽 목록에서 분석할 변수를 선택하고 \'분석 실행\' 버튼을 누르세요.',
    'save_success': '결과가 성공적으로 저장되었습니다:',
    'analysis_done': '분석이 완료되었습니다.',
    'graph_saved': '그래프가 성공적으로 저장되었습니다.',
    'validation_passed': '데이터 검증이 통과되었습니다.',
    
    # 에러 메시지
    'file_error': '파일을 읽는 중 오류가 발생했습니다:',
    'select_error': '분석할 변수를 하나 이상 선택해주세요.',
    'data_error': '먼저 파일을 불러오세요.',
    'graph_error': 'Graph 목록에 변수를 추가하세요.',
    'save_error': '저장할 변수를 하나 이상 선택해주세요.',
    'save_fail': '파일을 저장하는 중 오류가 발생했습니다:',
    'not_numeric': '숫자 데이터가 아니므로 통계를 계산할 수 없습니다.',
    'var_not_exist': '해당 파일에 존재하지 않습니다.',
    'file_too_large': '파일이 너무 큽니다.',
    'invalid_file_format': '지원하지 않는 파일 형식입니다.',
    'insufficient_data': '분석에 필요한 최소 데이터가 부족합니다.',
    'invalid_gait_data': '유효한 보행 데이터를 찾을 수 없습니다.',
    'memory_error': '메모리가 부족합니다. 더 작은 파일을 사용해주세요.',
    
    # 진행 상태 메시지
    'analysis_progress': '분석 중... 잠시만 기다려주세요.',
    'loading_file': '파일을 로딩하는 중...',
    'validating_data': '데이터를 검증하는 중...',
    'calculating_stats': '통계를 계산하는 중...',
    'analyzing_gait': '보행 패턴을 분석하는 중...',
    'generating_plots': '그래프를 생성하는 중...',
    'saving_results': '결과를 저장하는 중...',
    
    # 데이터 품질 관련
    'data_quality_good': '데이터 품질: 양호',
    'data_quality_fair': '데이터 품질: 보통',
    'data_quality_poor': '데이터 품질: 불량',
    'null_percentage': '결측값 비율',
    'outliers_detected': '이상값 감지됨',
    'data_points_count': '데이터 포인트 수',
    
    # 보행 분석 관련
    'gait_cycles_found': '보행 주기 발견',
    'cycle_length_avg': '평균 주기 길이',
    'cycle_variability': '주기 변동성',
    'state_transitions': '상태 전환',
    'phase_statistics': '단계별 통계',
    
    # 통계 옵션
    'remove_outliers': '이상값 제거',
    'confidence_interval': '신뢰구간',
    'sample_statistics': '표본 통계',
    'population_statistics': '모집단 통계',
    'detailed_analysis': '상세 분석',
    
    # 파일 관련
    'header_row_ask': '데이터가 시작되는 행 번호를 입력하세요 (첫 행=1):',
    'header_row_title': '헤더 위치 확인',
    'save_title': '결과 저장',
    'csv_file': 'CSV 파일',
    'all_file': '모든 파일',
    'select_folder': '폴더 선택',
    'file_encoding_detected': '파일 인코딩 감지됨',
    
    # 검증 메시지
    'validation_warnings': '검증 경고',
    'validation_errors': '검증 오류',
    'data_summary': '데이터 요약',
    'quality_report': '품질 보고서',
    
    # 새로운 기능들
    'auto_detect_header': '헤더 자동 감지',
    'smart_outlier_detection': '스마트 이상값 감지',
    'enhanced_statistics': '향상된 통계 분석',
    'robust_gait_analysis': '강화된 보행 분석',
    'export_options': '내보내기 옵션',
    'import_settings': '설정 가져오기',
    'export_settings': '설정 내보내기',
    'reset_settings': '설정 초기화'
}

LANG_EN = {
    # Basic GUI
    'app_title': 'Motion Data Analyzer v3.0',
    'file_not_selected': 'No file selected.',
    'select_file': 'File to analyze:',
    'open_file': 'Open File',
    'var_select': 'Variable Selection',
    'select_list': 'Selected List',
    'add_to_select': '→ Add to List',
    'remove_from_select': 'Remove from List',
    'graph_list': 'Graph List',
    'add_to_graph': '→ Add to Graph',
    'remove_from_graph': 'Remove from Graph',
    'draw_graph': 'View Graph',
    'draw_norm_graph': 'View Normalized Graph',
    'save_graph': 'Save Graph',
    'save_norm_graph': 'Save Normalized Graph',
    'analysis_result': 'Analysis Results',
    'run_analysis': 'Run Analysis',
    'save_result': 'Save Results',
    'data_quality': 'Data Quality',
    'statistics_options': 'Statistics Options',
    
    # Success messages
    'file_loaded': 'File loaded successfully.\nSelect variables from the left list and click \'Run Analysis\'.',
    'save_success': 'Results saved successfully:',
    'analysis_done': 'Analysis completed.',
    'graph_saved': 'Graph saved successfully.',
    'validation_passed': 'Data validation passed.',
    
    # Error messages
    'file_error': 'Error reading file:',
    'select_error': 'Please select at least one variable for analysis.',
    'data_error': 'Please load a file first.',
    'graph_error': 'Please add variables to the graph list.',
    'save_error': 'Please select at least one variable to save.',
    'save_fail': 'Error saving file:',
    'not_numeric': 'Cannot calculate statistics for non-numeric data.',
    'var_not_exist': 'Variable does not exist in the file.',
    'file_too_large': 'File is too large.',
    'invalid_file_format': 'Unsupported file format.',
    'insufficient_data': 'Insufficient data for analysis.',
    'invalid_gait_data': 'No valid gait data found.',
    'memory_error': 'Insufficient memory. Please use a smaller file.',
    
    # Progress messages
    'analysis_progress': 'Analyzing... Please wait.',
    'loading_file': 'Loading file...',
    'validating_data': 'Validating data...',
    'calculating_stats': 'Calculating statistics...',
    'analyzing_gait': 'Analyzing gait patterns...',
    'generating_plots': 'Generating plots...',
    'saving_results': 'Saving results...'
}


def get_language(lang_code='ko'):
    """언어 설정 반환"""
    languages = {
        'ko': LANG_KO,
        'en': LANG_EN
    }
    return languages.get(lang_code, LANG_KO)


def get_message(key, lang_code='ko', default=None):
    """특정 메시지 반환"""
    lang = get_language(lang_code)
    return lang.get(key, default or key)


def format_message(key, lang_code='ko', **kwargs):
    """포매팅된 메시지 반환"""
    message = get_message(key, lang_code)
    try:
        return message.format(**kwargs)
    except (KeyError, ValueError):
        return message
