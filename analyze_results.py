#!/usr/bin/env python3
"""
IcarusGym 결과 분석 도구
pickle 파일을 다양한 형식으로 변환하고 시각화하는 스크립트
"""

import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime

def load_pickle_results(pickle_path):
    """pickle 파일 로드"""
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def extract_data_dict(results):
    """ResultSet에서 분석 가능한 딕셔너리 추출"""
    data = []
    
    for i, result_tuple in enumerate(results):
        config, metrics = result_tuple
        
        # 실험 설정 추출
        experiment_data = {
            'experiment_id': i + 1,
            'timestamp': datetime.now().isoformat(),
            
            # 토폴로지 정보
            'topology_name': config['topology']['name'],
            'topology_delay': config['topology']['delay'],
            
            # 워크로드 정보
            'workload_name': config['workload']['name'],
            'n_contents': config['workload']['n_contents'],
            'n_warmup': config['workload']['n_warmup'],
            'n_measured': config['workload']['n_measured'],
            'request_rate': config['workload']['rate'],
            'alpha': config['workload']['alpha'],
            
            # 배치 전략
            'content_placement': config['content_placement']['name'],
            'cache_placement': config['cache_placement']['name'],
            'network_cache_ratio': config['cache_placement']['network_cache'],
            
            # 캐싱 전략
            'strategy_name': config['strategy']['name'],
            'strategy_content_max': config['strategy']['content_max'],
            'decision_interval': config['strategy']['decision_interval'],
            
            # 캐시 정책
            'cache_policy': config['cache_policy']['name'],
            'cache_policy_content_max': config['cache_policy']['content_max'],
            
            # 설명
            'description': config['desc'],
            
            # 성능 메트릭
            'cache_hit_ratio': metrics['CACHE_HIT_RATIO']['MEAN'],
        }
        
        # 노드별 성능 추가
        per_node_cache = metrics['CACHE_HIT_RATIO']['PER_NODE_CACHE_HIT_RATIO']
        for node, ratio in per_node_cache.items():
            experiment_data[f'node_{node}_cache_hit_ratio'] = ratio
            
        per_node_server = metrics['CACHE_HIT_RATIO']['PER_NODE_SERVER_HIT_RATIO']
        for node, ratio in per_node_server.items():
            experiment_data[f'node_{node}_server_hit_ratio'] = ratio
            
        data.append(experiment_data)
    
    return data

def save_to_json(data, output_path):
    """JSON 형식으로 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON 저장 완료: {output_path}")

def save_to_csv(data, output_path):
    """CSV 형식으로 저장"""
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✅ CSV 저장 완료: {output_path}")

def create_visualization(data, output_dir):
    """시각화 생성"""
    df = pd.DataFrame(data)
    
    # 1. 캐시 히트율 막대 그래프
    plt.figure(figsize=(10, 6))
    plt.bar(df['experiment_id'], df['cache_hit_ratio'] * 100)
    plt.title('Cache Hit Ratio by Experiment')
    plt.xlabel('Experiment ID')
    plt.ylabel('Cache Hit Ratio (%)')
    plt.ylim(0, 105)
    
    # 값 표시
    for i, v in enumerate(df['cache_hit_ratio'] * 100):
        plt.text(i + 1, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    hit_ratio_path = Path(output_dir) / 'cache_hit_ratio.png'
    plt.savefig(hit_ratio_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 히트율 그래프 저장: {hit_ratio_path}")
    
    # 2. 실험 설정 요약 테이블
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 요약 테이블 데이터
    summary_data = []
    for _, row in df.iterrows():
        summary_data.append([
            f"실험 {row['experiment_id']}",
            row['strategy_name'],
            row['cache_policy'],
            f"{row['cache_hit_ratio']*100:.1f}%",
            row['n_contents'],
            row['n_measured']
        ])
    
    table = ax.table(cellText=summary_data,
                    colLabels=['실험', '전략', '캐시 정책', '히트율', '컨텐츠 수', '측정 요청'],
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title('Experiment Summary', pad=20, fontsize=14, fontweight='bold')
    
    summary_path = Path(output_dir) / 'experiment_summary.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 요약 테이블 저장: {summary_path}")

def print_analysis_summary(data):
    """분석 결과 요약 출력"""
    df = pd.DataFrame(data)
    
    print("\n" + "="*60)
    print("🔍 IcarusGym 결과 분석 요약")
    print("="*60)
    
    print(f"📊 총 실험 수: {len(df)}")
    print(f"🎯 평균 캐시 히트율: {df['cache_hit_ratio'].mean()*100:.2f}%")
    print(f"🏆 최고 캐시 히트율: {df['cache_hit_ratio'].max()*100:.2f}%")
    print(f"📉 최저 캐시 히트율: {df['cache_hit_ratio'].min()*100:.2f}%")
    
    print(f"\n🔧 사용된 전략:")
    strategies = df['strategy_name'].unique()
    for strategy in strategies:
        count = len(df[df['strategy_name'] == strategy])
        avg_hit = df[df['strategy_name'] == strategy]['cache_hit_ratio'].mean() * 100
        print(f"   - {strategy}: {count}개 실험, 평균 {avg_hit:.2f}% 히트율")
    
    print(f"\n📝 사용된 캐시 정책:")
    policies = df['cache_policy'].unique()
    for policy in policies:
        count = len(df[df['cache_policy'] == policy])
        avg_hit = df[df['cache_policy'] == policy]['cache_hit_ratio'].mean() * 100
        print(f"   - {policy}: {count}개 실험, 평균 {avg_hit:.2f}% 히트율")

def main():
    parser = argparse.ArgumentParser(description='IcarusGym 결과 분석 도구')
    parser.add_argument('pickle_file', nargs='?', default='result.pickle',
                       help='분석할 pickle 파일 경로 (기본: result.pickle)')
    parser.add_argument('--output-dir', '-o', default='analysis_output',
                       help='출력 디렉토리 (기본: analysis_output)')
    parser.add_argument('--no-viz', action='store_true',
                       help='시각화 생성 건너뛰기')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # 결과 로드
        print(f"📂 pickle 파일 로드 중: {args.pickle_file}")
        results = load_pickle_results(args.pickle_file)
        
        # 데이터 추출
        print("🔄 데이터 추출 중...")
        data = extract_data_dict(results)
        
        # 다양한 형식으로 저장
        save_to_json(data, output_dir / 'results.json')
        save_to_csv(data, output_dir / 'results.csv')
        
        # 시각화 생성
        if not args.no_viz:
            print("📊 시각화 생성 중...")
            create_visualization(data, output_dir)
        
        # 요약 출력
        print_analysis_summary(data)
        
        print(f"\n✨ 분석 완료! 결과는 '{output_dir}' 디렉토리에 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
