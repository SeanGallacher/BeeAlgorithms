import time

import pandas as pd
import numpy as np
import cv2
import os
import moviepy.video.io.ImageSequenceClip
#import pydmd
import matplotlib.animation as animation
from IPython.display import HTML
from scipy import fftpack, signal
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN
from scipy.fftpack import fft2, ifft2
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
def convolutionMax(img, k):
    kernal = np.ones((k,k))
    kernal[k//2-k//4:k//2+k//4, k//2-k//4:k//2+k//4] +=1

    transform = signal.convolve2d(img, kernal, boundary='symm', mode='same')
    return np.max(transform)
def k_largest_index_argpartition_v2(a, k):
    idx = np.argpartition(a.ravel(),a.size-k)[-k:]
    return np.column_stack(np.unravel_index(idx, a.shape))
def dmd(x1,x2):
    u, s, vh = np.linalg.svd(x1, full_matrices=False)
    atil = u.T@x2@vh.T@np.linalg.inv(np.diag(s))
    evals, evecs = np.linalg.eig(atil)
    phi = x2@vh.T@np.linalg.inv(np.diag(s))@evecs
    phi = np.real(phi)
    return phi, s, vh
def createCircleMask(imageShape, radius):
    rows, cols = imageShape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.ones((rows, cols), np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]

    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius * radius

    mask[mask_area] = 0
    return mask
def writeVideoCV(heatmapImages, fps):


    width, height, _ = heatmapImages[0].shape
    video = cv2.VideoWriter('heatmapVideo2.mp4',  cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for img in heatmapImages:
        video.write(img.astype(np.uint8))
    video.release()
def writeVideoPNG(path, folderName):
    imageFolder = f'{path}heatMapVideos/'
    image_files = [os.path.join(imageFolder, img)
                   for img in os.listdir(imageFolder)
                   if img.endswith(".png")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=30)
    clip.write_videofile(f'{path}{folderName}/heatmapVideo.mp4')
def displayAnimation(frames):
    fig, ax = plt.subplots()
    frames_list = []
    for n, img in enumerate(frames):
        im = ax.imshow(np.array(img))
        if n == 0:
            ax.imshow(np.array(img))
        frames_list.append([im])

    ani = animation.ArtistAnimation(fig, frames_list, interval=50, blit=True,
                                    repeat_delay=1000)

    HTML(ani.to_jshtml())
class BeeMotionTracker():
    def __init__(self, videoPath, locationPath, name,   Images = (0,100), b = 5):
        ## load in frames
        video = cv2.VideoCapture(videoPath)
        self.name = name
        success, image = video.read()



        self.frames = []
        self.heatmapImages = []

        count = 0
        while success:
            count +=1
            if count >= Images[0] and count < Images[1]:
                self.videoLoaded = success
                self.frames.append(image[:,:,2])
            success, image = video.read()
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        if int(major_ver) < 3:
            self.fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            self.fps = video.get(cv2.CAP_PROP_FPS)
        if self.videoLoaded:
            self.imageShape = self.frames[0].shape
            print(f'image shape : {self.imageShape}')
        self.vidLen = len(self.frames)
        self.locations = pd.read_csv(locationPath)
        self.locationCors = self.locations[['x','y']].to_numpy()
        self.waggles = self.locations[['dance_movement']].to_numpy()
        self.previousDifImages = []

        x = []
        for img in self.frames:
            x.append(img.flatten())
        X1 = x[:-1]
        X2 = x[1:]
        self.frameslength = b
        self.X = []
        for i in range(self.vidLen - self.frameslength):
            x1 = X1[i: i + self.frameslength]
            x2 = X2[i: i + self.frameslength]
            self.X.append((np.array(x1).T, np.array(x2).T))
        self.phis_all = []
        self.ss_all = []
        self.vhs_all = []
    def predictMotion(self, FFT_cof = 20):
        ## Create edge images through FFT then subtract there score
        #print('starting fft')
        dfts = [np.fft.fftshift(fft2(img)) for img in self.frames]
        #print('finsihing fft')
        mask = createCircleMask(self.imageShape, FFT_cof)
        plt.imshow(np.log(np.abs(dfts[0]))*mask, cmap='gray')
        plt.title(f'{self.imageShape} fft mask')
        plt.show()
        self.tranforms = [ifft2(np.fft.ifftshift(dft_shift*mask)).astype(float) for dft_shift in dfts]

        plt.imshow(self.tranforms[0], cmap='gray')
        plt.colorbar()
        plt.title(self.name + "transformation")
        plt.show()
        vars = [np.var(im) for im in self.tranforms]


        self.difImages = [np.abs(self.tranforms[i] - self.tranforms[i+1]) for i in range(len(self.tranforms)-1)]
        self.previousDifImages.append(self.difImages)

    def exportHeatMap(self, folderName, num_images = 1000):
        self.heatmapImages = []
        path = f'//'
        png = False
        for loc, img, count, pred in zip(self.locationCors[:num_images], self.difImages[:num_images],
                                   [i for i in range(num_images)], self.predictions):
            x_true = int(loc[0])
            y_true = int(loc[1])
            plt.scatter(x_true,y_true,s=100)
            x_pred = pred[:,0]
            y_pred = pred[:, 1]
            plt.scatter(x_pred, y_pred, s=100)

            plt.imshow(img, cmap='hot', interpolation='sinc')
            plt.colorbar()
            plt.title(self.name)

            if png:
                fileName = f'heatmapImage{count} heat map'
                plt.savefig(path+fileName)
                os.rename(path + fileName+ '.png', f'{path}{folderName}/{fileName}.png')


            fig = plt.figure()
            heatMap = mplfig_to_npimage(fig)
            self.heatmapImages.append(heatMap)
            plt.show()

            plt.clf()
            plt.close('all')

        #writeVideoCV(self.heatmapImages, 20)


        self.heatmapImages = self.heatmapImages

    def predict(self, boxesW=5, boxesH=5, k=5):
        regionPixelWidth, regionPixelHeight = int(self.imageShape[0]/boxesW), int(self.imageShape[1]/boxesH)
        accuracies = []
        for loc, img in zip(self.locationCors, self.difImages):
            scoreMatrix = np.zeros((boxesW, boxesW))
            for i in range(boxesH):
                for j in range(boxesW):
                    lowBound = i * regionPixelHeight
                    upBoud = (i + 1) * regionPixelHeight
                    leftBound = j * regionPixelWidth
                    rightBound = (j + 1) * regionPixelWidth
                    region = img[lowBound: upBoud, leftBound:rightBound]
                    if loc[0] >= lowBound and loc[0] < upBoud:
                        if loc[1] >= leftBound and loc[1] < rightBound:
                            correctBox = (i, j)
                    intesityScore = np.sum(region)
                    scoreMatrix[i, j] = intesityScore
            ind = k_largest_index_argpartition_v2(scoreMatrix, k)
            ind = [(i,j) for i,j in ind]
            if correctBox in ind:
                accuracies.append(1)
            else:
                accuracies.append(0)
        accuracy = sum(accuracies)/len(accuracies)
        accuraciesWaggle = []
        for i, a in enumerate(accuracies):
            if self.waggles[i] == 1:
                accuraciesWaggle.append(a)
        accuracyWaggle = sum(accuraciesWaggle)/len(accuraciesWaggle)
        print(f'accuracy : {accuracy}')
        print(f'accuracyWaggle : {accuracyWaggle}')
        return accuracy, accuracyWaggle
    def dotDetector(self, b, r, sr, th):
        pixelScores = []
        pixelActivations = []
        for t in range(b, self.vidLen):
            B = np.array(self.difImages[t-b:t])
            B_std = (B - np.min(B, axis=0)) / (np.max(B,axis=0) - np.min(B))
            B_scaled = B_std * (1 - -1) + -1

            score = lambda m: (B_scaled[m,:,:]*np.cos(2*np.pi*r * m/sr))**2 + (B_scaled[m,:,:]*np.sin(2*np.pi*r * m/sr))**2
            scores = np.zeros_like(B)
            for m in range(b):
                scores += score(m)
            scores /= b

            scores = np.where(scores >= th, 1, scores)
            scores = np.where(scores < th, 0, scores)

            pixelScores.append(scores)
            pixelActivations.append(zip(*np.where(scores==1)))
        return pixelScores, pixelActivations
    def berlinPredict(self, b,r,sr, th, cmin1, cmin2, dmax1, dmax2):
        st = time.time()
        DDMatrix, DDInd = self.dotDetector(b, r, sr, th)
        print(f'finsihed')
        predictedWaggleBeeCenters = []
        ## Get clusters of predicted waggle points to get the center of the bee
        for inds in DDInd:
            waggleCenters = []
            ##dmax1 should be half the lenght of the bee
            clustering = DBSCAN(eps=dmax1, min_samples=cmin1).fit(inds)
            labels = clustering.labels_
            for i in range(max(labels)):
                points_of_cluster = inds[labels == 0, :]
                centroid = np.mean(points_of_cluster, axis=0)
                print(centroid)
                waggleCenters.append(centroid)

            predictedWaggleBeeCenters.append(np.array(waggleCenters))

        ## create possible paths
        self.predictions = [np.zeros((1,2)) if i < b else predictedWaggleBeeCenters[i-b] for i in range(self.vidLen)]
        for centriods in predictedWaggleBeeCenters:
            pass

        falsePositives = []
        accuracy = []
        ## evaulate location accuracy
        for i, centriods in enumerate(predictedWaggleBeeCenters):
            if self.waggles[i+b] == 0 or len(centriods) == 0:
                falsePositives.append(len(centriods))
                continue
            print(len(centriods))
            distances = np.sqrt(np.sum(np.square(centriods - self.locationCors[i+b]), axis=0), axis=0)
            print(len(distances))
            distances[distances < dmax1*2] = 0
            ## if predicted center is within a bees lenght of actual center
            if np.sum(distances) > 0:
                accuracy.append(1)
            else:
                falsePositives.append(len(centriods))
                accuracy.append(0)
        print(f'Berlin accuracy: {np.mean(accuracy)}')
        return np.mean(accuracy)
    def computeDMD(self):
        self.phis_all = []
        self.ss_all = []
        self.vhs_all = []
        checkplotAlign = False
        for x1, x2 in self.X:
            phis, ss, vhs = dmd(x1, x2)
            self.phis_all.append(phis)
            self.ss_all.append(ss)
            self.vhs_all.append(vhs)
            imageIndex = self.frameslength
            if checkplotAlign:
                plt.title(f'index: {imageIndex}')
                plt.imshow(x2[:,-1].reshape(self.imageShape))
                plt.show()
                plt.title(f'index: {imageIndex}')
                plt.imshow(self.frames[imageIndex])
                plt.show()
                plt.imshow(x2[:,-1].reshape(self.imageShape)-self.frames[imageIndex])
                plt.colorbar(
                )
                plt.show()
            imageIndex += 1

    def dmdPredict(self,plot=False):
        fig, ax = plt.subplots()
        imageIndex = self.frameslength
        accuracies = []

        ## each pass through X is analysis for on image
        for phis, ss, vhs in zip(self.phis_all, self.ss_all, self.vhs_all):
            waggle_truth = self.waggles[imageIndex]
            waggle_pred = 0

            ## get dyamic mode decomp
            st = time.time()
            ## for the modes, check the frequency
            for phi, s, vh in zip(phis.T, ss, vhs):
                phi[np.abs(phi) < .008] = 0
                strengths = fftpack.fft(vh)
                freqs = fftpack.fftfreq(len(vh)) * self.fps
                strongestFreq = freqs[np.argmax(strengths)]
                hotSpot = convolutionMax(np.abs(phi).reshape(self.imageShape), 10)
                if (np.abs(strongestFreq) <= 16 and np.abs(strongestFreq) >= 10) and np.sum(np.abs(phi)) > 50:
                        #or convolutionMax(np.abs(p).reshape(self.imageShape), 10) > .9:
                    print(f'abs : {np.sum(np.abs(phi))}')
                    waggle_pred = 1
                    if plot:
                        plt.title(f'index: {imageIndex} s: {s} strongestFreq: {strongestFreq} \n hotspot: {hotSpot}')
                        plt.imshow(phi.reshape(self.imageShape))
                        ax.set(facecolor="white")
                        plt.colorbar()
                        plt.show()
                        if False:
                            plt.title(f'index: {imageIndex}')
                            plt.imshow(self.frames[imageIndex])
                            loc = self.locationCors[imageIndex]
                        x_true = int(loc[0])
                        y_true = int(loc[1])
                        if waggle_truth == 1:
                            color = 'red'
                        else:
                            color = 'blue'
                        plt.scatter(x_true, y_true, s=100, color = color)
                        plt.show()

            imageIndex +=1
            print(time.time() - st)
            accuracies.append(waggle_truth-waggle_pred)
        accuracies = np.array(accuracies)
        predLen = len(accuracies)
        correct = (accuracies == 0).sum()

        falsePos = (accuracies == -1).sum()
        falseNeg = (accuracies == 1).sum()
        waggleArray = np.array(self.waggles[self.frameslength:imageIndex])

        wagglePred = {"waggle_truth" : waggleArray.tolist(), "waggle_pred" : accuracies.tolist()}
        truePositives = (accuracies == 0)[np.where(waggleArray==1)].sum()
        truePositives = (accuracies == 0)[np.where(waggleArray == 0)].sum()
        df = pd.DataFrame.from_dict(wagglePred)
        df.to_csv(f'{PATH}wagglePredsFA.csv', index=True)
        print(f'DMD accuracy : {correct/predLen}')
        print(f'DMD precision:  {truePositives/(truePositives+falsePos)}')
        print(f'DMD recall : {truePositives/(truePositives+falseNeg)}')
        print(f'DMD FalsePos : {falsePos/predLen}')
        print(f'DMD FalseNeg {falseNeg/predLen}')

        return correct/predLen


def findBestCof():
    BMT = BeeMotionTracker(f'{DATA_PATH}Video/RawFootage/WaggleDance_1.mp4',
                           f'{DATA_PATH}Locations/WaggleDance_1_Locations.csv', name = 'waggle 1')
    cofs = [50, 150,200, 250, 300]
    accuracies = []
    for cof in cofs:
        BMT.predictMotion(cof)
        accuracies.append(BMT.predict())
    return cofs[np.argmax(accuracies)]

if __name__ == '__main__':
    DATA_PATH = '//beesearch-hand-annotated-data-main/'
    PATH = '//'
    n = 19
    print(f'Waggle dance {n}')
    BMT = BeeMotionTracker(f'{DATA_PATH}Video/RawFootage/WaggleDance_{n}.mp4',
                           f'{DATA_PATH}Locations/WaggleDance_{n}_Locations.csv', name=f'WaggleDance_{n}',
                           maxImages = 100)
    BMT.computeDMD()
    BMT.dmdPredict(plot=True)
    if False:
        for n in [19]:
            print(f'Waggle dance {n}')
            BMT = BeeMotionTracker(f'{DATA_PATH}Video/RawFootage/WaggleDance_{n}.mp4',
                                   f'{DATA_PATH}Locations/WaggleDance_{n}_Locations.csv', name = f'WaggleDance_{n}')
            if BMT.videoLoaded:
                if np.sum(BMT.imageShape) > 2000:
                    continue
                for factor in [3.5, 6]:
                    cof = np.sum(BMT.imageShape[0])//factor
                    BMT.predictMotion(cof)
                    BMT.predict(5,5)
                    BMT.berlinPredict(8, 13, 30, .9, 20, 5, 50, 100)

            else:
                print(f'Waggle dance {n} failed')

