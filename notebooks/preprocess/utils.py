import spacy
from spacy.matcher import Matcher
from spacy.tokens import Token, Doc
from spacy.language import Language



patterns = [[{"TEXT": {"REGEX": "<.*?>"}},],
[{"TEXT": {"REGEX": "<"}},
{"TEXT": {"REGEX": "h1>"}}]
]
nlp = spacy.blank('pt')
matcher = Matcher(nlp.vocab)
matcher.add("delimiter_token", patterns)

def get_sentence(token):
    """
     Gets the sentence where the answer is found.
    """

    for sent in token.doc.sents:
        if sent.start <= token.i <= sent.end:
            return sent
        

if not Doc.has_extension("questions"):
    Doc.set_extension("questions", default=[])

if not Doc.has_extension("answers"):
    Doc.set_extension("answers", default=[])

if not Doc.has_extension("parag_ans"):
    Doc.set_extension("parag_ans", default=[])

if not Doc.has_extension("answer_sentence"):
    Doc.set_extension("answer_sentence", default=[])

if not Doc.has_extension("sentences"):
    Doc.set_extension("sentences", default=[])

if not Doc.has_extension("paragraph_sentence"):
    Doc.set_extension("paragraph_sentence", default=[])

if not Token.has_extension("is_delimiter"):
    Token.set_extension("is_delimiter", default=False, force=True)

if not Token.has_extension("sent"):
    Token.set_extension('sent', getter=get_sentence, force=True)



def spacy_get_text_boundaries(text, context_doc):
    """
    Find the boundaries of the answers tokens in the context document
    text: string # answer in string format
    context_doc: spacy document # context where the answers boundaries must
                                #  be found
    """
    matcher = Matcher(nlp.vocab)

    pattern = [[{"ORTH": token.text} ]  for token in nlp(text)]
    matcher.add(text, pattern)
    matches = matcher(context_doc)
   
    answers_starts = []
    answers_ends = []
    for _, start, end in matches:
        answers_starts.append(start)
        answers_ends.append(end)

    return min(answers_starts), max(answers_ends)



        
def get_answer_sentence(token):
    """
     Gets the sentence where the answer is found.
    """

    for sent in token.doc.sents:
        if token._.is_delimiter:
            return sent

def remove_h1_token(text_doc):
    context_words = [token.text for token in text_doc if token.text != "<h1>"]

    spaces = [not token.is_punct for token in text_doc if token.text != "<h1>"]
    #start_span, end_span = start_span - num_h1, end_span - num_h1
    spaces.pop(0)
    spaces += [False]
    return Doc(nlp.vocab, words=context_words, spaces=spaces)

def spacy_mark_text_span(text_doc, start_span, end_span,):
  context_words = [token.text for token in text_doc if token.text != "<h1>"]
  num_h1 = (len(text_doc)) - len(context_words)
  spaces = [not token.is_punct for token in text_doc if token.text != "<h1>"]
  #start_span, end_span = start_span - num_h1, end_span - num_h1
  spaces.pop(0)
  spaces += [False]
  
  #FIX: punctuations are one index after where it should be
  spaces.insert(start_span,False)
  spaces.insert(end_span,False)
  
  context_words.insert(start_span, "<h1>")
  context_words.insert(end_span-num_h1, "<h1>")
 
  return Doc(nlp.vocab, words=context_words, spaces=spaces)


@Language.component("delimiter_tokenizer")
def delimiter_tokenizer(doc):
    matches = matcher(doc)
  
    for match_id, start, end in matches:
        for i in range(start, end):
            token = doc[i]
            token._.is_delimiter = True
            # Process the token if needed
            # For example, you can set a custom attribute on the token
        matched_span = doc[start: end] 
      
        with doc.retokenize() as retokenizer:
            retokenizer.merge(matched_span)
    return doc

@Language.component("answer_sentence")
def answer_sentence(doc):
    """
    Find the answer inside the paragragh using Ruled Based Matcher
    """

    answer_text = ""
    ans_found = False
    parag_sentence = None
    for token in doc:
        if ans_found and token.text != "<h1>":
            answer_text += token.text + " "
            
        if token._.is_delimiter :   
            
            
            if remove_h1_token(token._.sent) not in doc._.sentences:
                ans_found = True
                #- **Sentence Answer:** Sentence where the answer is found with the answer highlighted 
                #with `<h1> <h1>` tags. - Done
                doc._.answer_sentence.append(token._.sent)

                # **Sentence Answer:** Sentence where the answer 
                # is found with the answer highlighted with `<h1> <h1>` tags.
                doc._.sentences.append(remove_h1_token(token._.sent))

                #- **Paragraph Sentence**: The paragraph with the sentence
                # highlighted with `<h1> <h1>` tags. - DOne
                parag_sentence = spacy_mark_text_span(doc, 
                                                            token._.sent.start,
                                                            token._.sent.end)
                doc._.paragraph_sentence.append(parag_sentence)
            else:
                ans_found = False
                answer_text = ""
               
    return doc

# Add a computed property, which will be accessible as token._.sent
nlp.tokenizer.add_special_case("<h1>", [{"ORTH": "<h1>"}])

if "delimiter_tokenizer" not in nlp.pipe_names:
    nlp.add_pipe("delimiter_tokenizer", name="delimiter_tokenizer")
if 'sentencizer' not in nlp.pipe_names:
    nlp.add_pipe('sentencizer')
if "answer_sentence" not in nlp.pipe_names:
    nlp.add_pipe("answer_sentence", name="answer_sentence")

