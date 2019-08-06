
import os

if __name__ == "__main__":
    files = [f for f in os.listdir('video') if f.endswith('.mp4')]
    print('num_files: ' + str(len(files)))

    folder = 'cache'
    if not os.path.isdir(folder):
        os.makedirs(folder)

    print('building index...')
    i = 0
    frames = []
    for file in files:
        filename = os.path.join('video', file)
        file = file[3:]
        tokens = file.split('-')
        name = tokens[0] + '-' + tokens[1]
        print(name)