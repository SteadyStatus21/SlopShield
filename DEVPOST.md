## Inspiration

If you've scrolled Facebook with a grandparent lately, you've seen it: a little girl singing to soldiers, a celebrity "explaining" a crypto trick, a dog rescuing a baby from a river. None of it happened. Tools like Sora, Veo and Kling made fake video cheap, and scammers figured that out fast. The FBI's 2025 internet crime report put losses for Americans over 60 at $7.75 billion, up about 60% from the year before, with more than 3,100 senior complaints that specifically involved AI ([AARP's summary](https://www.aarp.org/money/scams-fraud/fbi-ftc-report-2025-losses.html), [HousingWire](https://www.housingwire.com/articles/fbi-seniors-cybercrime-2025/)).

Some platforms do label AI content, but most AI posts never get a label. We wanted something that sits quietly in the browser and taps you on the shoulder when a video is probably fake, without sending what you watch to anyone's server.

## What it does

SlopShield is a browser extension that checks the videos and images you scroll past on Instagram, Facebook, X, YouTube and pretty much any other site. When something looks AI-generated, it either puts a big red "Likely AI-generated" banner on it, or, if you picked blocking, it pauses it, blurs it and covers it with a warning. You can still watch it, but you have to click twice to get through, which is on purpose.

Everything runs on your own computer. There's an optional helper app that uses your laptop's graphics chip to make checks a lot faster, and when you install the extension, a setup page walks you through getting it running and asks whether you'd rather be warned or blocked.

If it gets something wrong, there are "Yes, AI" and "Not AI" buttons right on the post. Those clicks save examples, and running `train.bat` retrains the detector on them in about a minute. It only swaps in the new model if it actually scores better on a test set it has never seen, so a bad batch of labels can't quietly make it worse.

## How we built it

The extension is TypeScript (Chrome Manifest V3). A content script finds every image and video on the page, including ones hidden inside Reddit's shadow DOM, and watches which ones are actually on screen. The background worker takes a screenshot, crops out each video, trims black letterbox bars, and cuts tall videos into a few square tiles so the model sees the whole frame and not just the middle.

For the detector we started from a small vision transformer (ViT-S) from the [Community Forensics](https://huggingface.co/papers/2411.04125) project, which was trained on images from over 4,800 different generators. On top of it we trained our own small decision layer on about 19,000 crops: AI video from Veo, Kling and Sora, a few hundred phone-style AI shorts, real YouTube clips, real stock footage, photos, video game screenshots, meme-style text cards, and our own labelled scrolling. We also degraded a lot of the frames on purpose (shrinking, recompressing, adding captions) so they'd look like something that's been reposted five times.

Speed mattered because our test machine is a normal laptop with Intel integrated graphics. On the CPU in the browser a single crop takes about 1.6 seconds. WebGPU gets that to around 150 ms. The helper app runs the same model through Intel's OpenVINO and gets it down to about 47 ms. It only listens on your own machine and only talks to our extension, and if it isn't running, the extension just falls back to WebGPU.

To keep ourselves honest we wrote a benchmark script that scores every model exactly the way the extension does, on clips it never trained on. On those held-out sets the current model catches 54 of 60 modern AI videos, 207 of 241 phone-style AI shorts and 162 of 183 clips from the newest generators (Seedance 1.5, Kling 2.5). It wrongly flags 1 of 60 real videos, 13 of 198 real stock clips, and none of the 40 real photos or 30 game screenshots. On the posts we labelled from our own feeds, it catches roughly two out of three AI posts and wrongly flags about one in eleven real ones. That last number is the one we care about most, and it's the one we'd most like to improve.

## Challenges we ran into

Video games were the first big problem. Our original model thought rendered graphics were AI on every single frame and flagged 16 of 30 game screenshots. Switching base models and adding a second check got that to zero.

Instagram Reels fought us. When we paused a blocked reel, Instagram immediately pressed play again, and our code paused it again, thousands of times a second, until the browser locked up. Now we pause once, mute it and hide it instead of arguing with the page.

Our own UI kept ending up in the model's input. A debug label inside a screenshot crop, or the black padding around an Instagram carousel, was enough to cause false alarms. The capture area now skips our labels, and flat borders get trimmed before anything is scored.

GPU drivers lied to us. One backend (DirectML on the Intel chip) returned the exact same score for every image, so we checked every backend against plain CPU output before trusting it.

And captions turned out to be useless as evidence. Real videos say "AI" in the caption as a joke, and plenty of AI videos have no caption at all, so only real platform labels like "Made with AI" are trusted without the vision model.

## Accomplishments that we're proud of

It runs on a regular laptop with no cloud at all, it's fast enough to keep up with scrolling, and it keeps game footage and real photos clean while catching most of the AI video we threw at it. We're also proud of the feedback loop: label something while you scroll, retrain in a minute, and the new model only goes live if the numbers say it's better. Every number in this writeup comes out of that benchmark script, not from picking good demo clips.

## What we learned

Mostly, that you have to test on the stuff your users actually see. We got a much bigger detector (Smaug, about 20 times our model's size) running on the laptop as a second opinion. It looked great on public benchmarks, but on posts from our own feeds it did worse than flipping a coin, so we dropped it. A safety rule we added early, where both models had to agree before flagging anything, also turned out to be blocking real catches once our training data got better. Loosening it took held-out phone-style AI from 62% caught to 86% without adding false alarms on photos or games.

We also learned that single frames only get you so far. A model looking at one frame can't notice a spoon teleporting into a pan.

## What's next

- Looking at motion across frames, since that's where a lot of AI video still gives itself away.
- More real-world training data, especially ordinary-looking phone videos, which are still the hardest case.
- A parent mode with a PIN, per-site allowlists, and warnings on the remote-access and "tech support" pages that scammers send people to.
- A mobile version.
