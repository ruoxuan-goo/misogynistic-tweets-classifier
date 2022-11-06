import re
import html


class CleanTweets:

    def __init__(self):
        self.BOS = 'xbos'  # beginning-of-sentence tag
        self.FLD = 'xfld'  # data field tag
        self.XNEL = 'xnel'  # non english occurence
        self.XURL = 'xurl'  # url occurence
        self.XATP = 'xatp'  # @Person occurance
        self.XRTU = 'xrtu:'  # retweet unmodified
        self.XRTM = 'xrtm:'  # retweet modified

        self.re1 = re.compile(r'  +')

    def cleanTweet(self, df_train, text_column):
        text_PP = []

        for idx in df_train.index:
            sent = df_train.loc[idx, text_column]
            sent = self.prepro_erroroneus(sent)
            text_PP.append(sent)

        df_train['processed_data'] = text_PP
        return df_train

    def cleanNonASCII(self, df_train, text_column):
        text_PP = []

        for idx in df_train.index:
            sent = df_train.loc[idx, text_column]
            sent = self.prepro_nonASCII(sent)
            text_PP.append(sent)

        df_train['processed_data'] = text_PP
        return df_train

    def prepro_nonASCII(self, text):
        '''
        Remove Non ASCII characters from the dataset. Naive way to remove non english
        Arguments:
            text: str
        returns: 
            text: str
        '''
        return ''.join(i for i in text if ord(i) < 128)

    def prepro_erroroneus(self, tweet):

        def fixup(x):
            """ Cleans up erroroneus characters"""
            x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
                'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
                '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
                ' @-@ ', '-').replace('\\', ' \\ ').replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').replace('rt @', '@')
            return self.re1.sub(' ', html.unescape(x))

        tweet = fixup(tweet)
        tweet = re.sub(r'^RT @ \w+:', self.XRTU, tweet)
        tweet = re.sub(r'^MRT @ \w+:', self.XRTM, tweet)
        tweet = re.sub(r'@ \w+', self.XATP, tweet)
        tweet = re.sub(r'http\S+', self.XURL, tweet)
        tweet = re.sub(r'(.)\1+', r'\1\1', tweet).strip()
        tweet = tweet.lower()
        return tweet
