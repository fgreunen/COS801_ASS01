import jsonpickle
class ResultsManager:
    def save(self, filename, obj):
        filename = 'RunData/' + filename
        fileObject = open(filename,'w') 
        fileObject.write(jsonpickle.encode(obj))
        fileObject.close() 
    def load(self, filename):
        filename = 'RunData/' + filename
        fileObject = open(filename,'r')  
        b = jsonpickle.decode(fileObject.read())
        fileObject.close()
        return b