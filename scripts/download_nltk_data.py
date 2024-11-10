import nltk

def download_nltk_resources():
    """下载必要的NLTK资源"""
    resources = [
        'stopwords',
        'punkt',
        'averaged_perceptron_tagger'
    ]
    
    for resource in resources:
        print(f"下载 {resource}...")
        nltk.download(resource)

if __name__ == "__main__":
    download_nltk_resources() 