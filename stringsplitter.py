#Split strings by capital letter
def string_splitter(word, glue):
  """take a word with a capital initial and split 
  it by the capital letters appering in the word 
  and paste them with glue""" 
  return glue.join([letter + rest for letter, rest in zip(re.findall(r'[A-Z]', word),re.split(r'[A-Z]', word)[1:])])

if __name__ == "__main__":
  print(string_splitter("MultipleWordString", " "))
  


