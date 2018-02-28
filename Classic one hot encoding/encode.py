#Classic one hot encpoding
#letter universe : english alphabets + space

from numpy import argmax
# define input string
data = 'hello world'
print(data)
# define universe of possible input values
alphabet = 'abcdefghijklmnopqrstuvwxyz '
print(alphabet)


# define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

print("Char to int ")
print(char_to_int)


print("Int to Char ")
print(int_to_char)

print(type(char_to_int))


# integer encode input data
integer_encoded = [char_to_int[char] for char in data]
print(integer_encoded)

print(len(alphabet))

onehot_encoded = list()
for value in integer_encoded:
    letter = [0 for _ in range(len(alphabet))]
    letter[value]= 1
    # print(letter)
    onehot_encoded.append(letter)

print("final one hot encoded ")
print(onehot_encoded)



# invert encoding
inverted = int_to_char[argmax(onehot_encoded[0])]+int_to_char[argmax(onehot_encoded[1])]
print(inverted)