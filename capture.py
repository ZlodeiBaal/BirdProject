import cv2
import time
import caffe
import numpy as np
from scipy import misc
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, InlineQueryHandler
from telegram import InlineQueryResultCachedPhoto
import logging
from uuid import uuid4
import os
from time import sleep
import datetime

#Start command
def start(bot, update):
    update.message.reply_text('Hi, '+update.message.from_user.first_name+'. Here we observe birds on window. Just type /help for all comands')
    update.message.reply_text('We work only from 8.00 till 18.00 on Moscow time (GMT+3)')

#Used on new foto inc
def alarm():
    print "Alarm!"
    d = datetime.datetime.now()
    if ((d.hour+d.minute/60.0>7.7)and(d.hour<18)):
        global THISBOT
        for chat_id in ChatArray:
            try: 
                photo_file = open(last_adress, 'rb')
                THISBOT.sendPhoto(chat_id, photo_file)
            except:
                print "User delete himself"
                ChatArray.remove(chat_id )
                RewriteChat()
        global ListOf
        photo_file = open(last_adress, 'rb')
        id = THISBOT.sendPhoto("@win_feed", photo_file)
#Let's save image names. Easy way to read history.
        ListOfImg.append(id.photo[0].file_id)
        if (len(ListOfImg)>=6):
            del ListOfImg [0]
            
#Help comand
def help(bot, update):
    update.message.reply_text('We have few command here:')
    update.message.reply_text('/lastgood - send you last good photo')
    update.message.reply_text('/startspam - you want all the photo!')
    update.message.reply_text('/stopspam - STOP IT!')
    update.message.reply_text('/howmuch - How much birds came today')
    update.message.reply_text('/start - Just hi!')
    update.message.reply_text('/help - Yep, we here')

#Say how much photo was taking during work
def HowMuch(bot, update):
    global CountIt
    update.message.reply_text("I have seen "+str(CountIt)+" birds today (or from my restart)")

#Last photo with bird
def LastGood(bot, update):
    photo_file = open(last_adress, 'rb')
    update.message.reply_photo(photo_file)

#Save list of users
def RewriteChat():
    if (os.path.isfile('SpamAdress.txt')):

        os.remove('SpamAdress.txt')
    fw = open('SpamAdress.txt', 'a')
    for chat_id in ChatArray:
        fw.write(str(chat_id) + '\n')
    fw.close()

#Some strange ideas
#Inline image input
#History search
def inlinequery(bot, update):
    query = update.inline_query.query
    results = list()
    for i in range(len(ListOfImg)):
        results.append(InlineQueryResultCachedPhoto(id=uuid4(),title=str(i),
                                            photo_file_id=ListOfImg[i]))
    update.inline_query.answer(results)

#Add new user
def StartSpam(bot, update):
    chat_id = update.message.chat_id
    if (len(ChatArray)<15):
        if (chat_id not in ChatArray):
            ChatArray.append(chat_id)
        update.message.reply_text("To stop spam just /StopSpam")
        RewriteChat()
    else:
        update.message.reply_text("Sorry! Spam is closed due hight amount of users...")

#Remove user from list
def StopSpam(bot, update):
    chat_id = update.message.chat_id
    try:
        ChatArray.remove(chat_id)
        update.message.reply_text("No more spam 4 u!")
        RewriteChat()
    except ValueError:
        update.message.reply_text("U lie 2 me!")

#erroor workout
def error(bot, update, error):
    logger.warn('Update "%s" caused error "%s"' % (update, error))

#Init Caffe net + caffe transformer
net = caffe.Net('deploy.prototxt', 'SQ.caffemodel', caffe.TEST)
net.blobs['data'].reshape(1, 3,  227, 227)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
mu = np.array([128.0, 128.0, 128.0])
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', mu)

#Bot initiation
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
updater = Updater("SECRET")
global THISBOT
THISBOT = updater.bot
global ChatArray  
ChatArray  = []
global ListOfImg
ListOfImg = []
dp = updater.dispatcher
global CountIt
CountIt = 0
dp.add_handler(CommandHandler("start", start))
dp.add_handler(CommandHandler("help", help))
dp.add_handler(CommandHandler("lastgood", LastGood))
dp.add_handler(CommandHandler("startspam", StartSpam))
dp.add_handler(CommandHandler("howmuch", HowMuch))
dp.add_handler(CommandHandler("stopspam", StopSpam))
dp.add_handler(InlineQueryHandler(inlinequery))
dp.add_error_handler(error)
updater.start_polling()
global last_adress
last_adress="5_2.jpg" #Photo for first shoot
global BotCycle
BotCycle = True
#init list of users
if (os.path.isfile('SpamAdress.txt')):
    with open('SpamAdress.txt') as f:
        for line in f:
            ChatArray.append(int(line))




#init camera on RPI
video_capture = cv2.VideoCapture(0)
video_capture.set(3,1280)
video_capture.set(4,720)
video_capture.set(10, 0.6)
ret, frame_old = video_capture.read()
i=0
j=0
k=0


#Main circle:
#   If it's daytime
#   Once in 0.1 second
#   Chek if we see movement
#   if movement:
#     Chek picture by neural network
#     if bird:
#       send alarm
while BotCycle:
    time.sleep(0.1)
    dt = datetime.datetime.now()
    if ((dt.hour+dt.minute/60.0>7.7)and(dt.hour<18)):
        ret, frame = video_capture.read()
        diffimg = cv2.absdiff(frame, frame_old)
        d_s = cv2.sumElems(diffimg)
        d = (d_s[0]+d_s[1]+d_s[2])/(1280*720)
        frame_old=frame
        print d
        if i>30:
            if (d>14):
                frame = frame[:, :, [2, 1, 0]]
	        transformed_image = transformer.preprocess('data', frame)
                net.blobs['data'].data[0] = transformed_image
                net.forward()
                if (net.blobs['pool10'].data[0].argmax()!=0):
                    last_adress="base/"+str(j)+"_"+str(net.blobs['pool10_Q'].data[0].argmax())+".jpg"
                    misc.imsave(last_adress,frame)
                    alarm()
                    CountIt=CountIt+1
                    j=j+1
                else:
                    misc.imsave("base_d/"+str(k)+".jpg",frame)
                    k=k+1
        else:
            i=i+1
    else:
        time.sleep(1)
        CountIt=0
updater.stop()