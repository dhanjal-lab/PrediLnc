from app import app

if __name__ == 'main':
    app.run()


# gunicorn -w 2 -b 0.0.0.0:8085 wsgi:app
# nohup gunicorn -w 2 -b 127.0.0.1:8085 wsgi:app > gunicorn.log 2>&1 &