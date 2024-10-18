from Vectors import *

@ti.func 
def sort(field):
    '''
    Sort the field
    '''
    for i in field: 
        minIndex = i 
        for j in ti.ndrange((i, field.shape[0])):
            if field[j] < field[minIndex]:
                minIndex = j 
        field[i], field[minIndex] = field[minIndex], field[i]

newField = ti.field(int, shape = (5,))
newField[0] = 5 
newField[1] = 2 
newField[2] = 4 
newField[3] = -1
newField[4] = 10

@ti.kernel 
def testSort(newField: ti.template()): #type: ignore 
    sort(newField)
    for i in newField:
        print(newField[i])