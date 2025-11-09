_________________________________________________________________________________
Adriana R. Florez (adrirflorez@gmail.com / @rodrigad), 5/12/2024

Before writing a short commentary, I will explain a bit how the output.txt looks
(because I might have or might have not changed a bit the overall aesthetic
of the results so it would be easier for me to see.)

- System prompt
(for all response prompts... which are 3)
    - Description of the response prompt
    (for all generation configs... which are 5 or 6)
        - Description of the generation config
        - Next 3 elements are the outputs provided by the model, limited to 90 tokens

The thing about the generation configs is that on my test.py you can see all the ones 
that I tried when executing the full-fledged thing (which I did on Google Colab
because my laptop could not take it anymore> 
https://colab.research.google.com/drive/1FpW505FL_DA_tyAwdkhVRPZQ8UYe61Dh)

However, seeing that the results truly did not differ much and they were virtually giving me
the same things for each, I basically give you the output with just one generation config
so you can easily see how the model responds to each prompt and response spec. 
I was more interested on how the model reacted to the different prompts, which is what was
easier to test (rather than seeing how the output different when I changed the temperature
or number of beams in the config...)

_________________________________________________________________________________


COMMENTARY>

If we go over the different response prompt types (general, formatted and friendly), as well as 
the different requirements specified the user, I would say it works quite well - though with
improvements to point out!

If we look at the case of the general response, it does not avoid saying specific names,
which it was specifically asked not to. However, it also includes key ideas that were mentioned
by the user (i.e. accessibility in Lisbon accommodation, or special dietary needsa to take into account). 

The formatted approach worked just fine, following the format that was established in the response prompt. I think it would be very useful for users who want to get straight to the point and not
have to put up with the endless chatter that some language models provide.

I wanted to try out a nice response type, mimicking an actual human being that is friendly and warm
towards the client. It turns out super realistic in that way, and it does give you the feel of
actually interacting with an old woman travel agent who wants you to have fun in your trip.
The fact that it also greets you in the language of the place or of the user (I was getting some
Shabbat Shaloms as well) is also super cute and a nice addition.

In conclusion, although it might forget about some points mentioned in the response prompt,
the model really delivers in terms of what it is asked to do. I will reiterate again that the other generation configs gave me super similar response types, and that these ones serve as a very decent sample of what kind of outputs I was getting!