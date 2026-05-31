
import runpy, sys
sys.argv = ['sentiment_analysis_model.py', '--data', r'/mnt/data/sentiment_project/demo_sentiment_data.csv', '--model-out', r'/mnt/data/sentiment_project/sentiment_model.joblib']
runpy.run_path(r'/mnt/data/sentiment_project/sentiment_analysis_model.py', run_name='__main__')
