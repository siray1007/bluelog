from bluelog import create_app
import sys

sys.path.append('/app')

app = create_app()

if __name__ == '__main__':
    app.run(host="0.0.0.0")