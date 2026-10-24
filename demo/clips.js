// Demo feed contents. File name prefix is the ground truth used by scripts/eval.py:
// ai_*.mp4 = AI-generated, real_*.mp4 = real footage.
// Mix of cases: AI with a platform label, AI with only hashtags, AI with no text cues
// (vision has to catch it), and real clips, including one with a misleading caption.
window.CLIPS = [
  { file: 'real_01.mp4', author: 'Linda Park', caption: 'Morning walk with Biscuit 🐶 the leaves are finally turning!' },
  { file: 'ai_01.mp4', author: 'Faith & Blessings Daily', caption: 'This little girl sang for the soldiers 😭 Type AMEN if this touched you 🙏', platformLabel: 'AI info' },
  { file: 'real_02.mp4', author: 'Tom Okafor', caption: 'Grandkids beat me at bowling again. Never getting a rematch.' },
  { file: 'ai_02.mp4', author: 'Nature Wonders', caption: 'Rare footage of a snow leopard rescuing a lost cub #sora #aivideo' },
  { file: 'real_03.mp4', author: 'Westside Farmers Market', caption: 'Apple cider donuts are back this Saturday 🍩' },
  { file: 'ai_03.mp4', author: 'Breaking News 24/7', caption: 'Shocking: officials caught on camera. Share before they delete this!' },
  { file: 'real_04.mp4', author: 'Maria Gonzalez', caption: 'So proud of my nephew at his first recital 🎻' },
  { file: 'ai_04.mp4', author: 'Crypto Wealth Tips', caption: 'Elon explains how to double your savings — link in comments 💰' },
  { file: 'real_05.mp4', author: 'Ridge Hiking Club', caption: 'Sunrise from the summit. You had to be there. (no, this is not AI lol)' },
  { file: 'ai_05.mp4', author: 'Cute Animals World', caption: 'Kitten and baby duck best friends 🥹 #ai #klingai' },
];
