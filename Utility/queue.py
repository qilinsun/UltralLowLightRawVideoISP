class Queue:
    """
    First-in-First-out queue with pre-defined bucket number
    
    """
    def __init__(self,bucketSize=16):
        self.memory = []
        self.bucketSize = bucketSize
        
    def enqueue(self, x):
        """
        Return front if queue is full
        Return injection if queue is not full
        """
        if len(self.memory) == self.bucketSize:
            self.memory.insert(0, x)
            return self.memory.pop()
        else:
            self.memory.insert(0, x)
            return x
        
    def dequeue(self):
        return self.memory.pop()
    
    def isEmpty(self):
        return len(self.memory) == 0
    
    def getFront(self):
        return self.memory[-1]
    
    def getRear(self):
        return self.memory[0]
    
    def getIndex(self,idx):
        return self.memory[idx]
    
    def __len__(self):
        return len(self.memory)

if __name__ == "__main__":
    queue = Queue()
    queue.enqueue(1)
    queue.enqueue("x")
    queue.enqueue(3.)
    print('>>> queue.queue[0]  ',queue.queue[0])
    print('>>> queue.queue[1]  ',queue.queue[1])
    print('>>> queue.queue[2]  ',queue.queue[2])
    
    """
    >>> queue.queue[0]   3.0
    >>> queue.queue[1]   x
    >>> queue.queue[2]   1
    """