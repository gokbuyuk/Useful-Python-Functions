def splitncap(str, splitter = "_"):
  """takes a string and a splitter, 
     splits the string with the splitter
     and capitalizes the first word""" 
  x = str.split(splitter)
  return(" ".join(x).capitalize())

if __name__ == "__main__":
  print(splitncap("name_surname", "_"))
  print(splitncap("name surname", " "))
