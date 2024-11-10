import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_resources():
    """下载必要的NLTK资源"""
    resources = [
        'stopwords',
        'punkt',
        'averaged_perceptron_tagger',
        'words'
    ]
    
    for resource in resources:
        print(f"正在下载 {resource}...")
        nltk.download(resource)

if __name__ == "__main__":
    download_nltk_resources() 