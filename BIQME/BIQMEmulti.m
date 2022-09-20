function pred = BIQMEmulti(numSteps)
  for i = 1:32
      im = imread(strcat(num2str(i-1), '.png'));
      pred(i) = BIQME(im);
  end
end
