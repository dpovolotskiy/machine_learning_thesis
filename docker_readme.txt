Сборка образа из Dockerfile
docker build -t image_capt:0.2.6 --force-rm /home/dpovolotskiy/Documents/git/machine_learning_thesis/Docker/
Добавление образа в DockerHub
docker login
docker tag image_capt:0.2.6 dpovolotskiy/image_caption_app:0.2.6
docker push dpovolotskiy/image_caption_app:0.2.6
Для запуска контейнера из DockerHub
docker run -i -d -p 5000:5000 dpovolotskiy/image_caption_app:0.2.6