with open ('drseuss.txt', 'r+', encoding="utf-8", errors='ignore') as f:
     with open ('drseuss_new.txt', 'w+') as nf:
        lines = f.readlines()
        replacements = [('\n', ''), ('   ', ' '), ('  ', ' '), ('\"', ''), ('...', '.'), ('....', '.'), ('?', '.'), ('!', '.')]
        lines = ' '.join(lines).lower()
        for current, new in replacements:
            lines = lines.replace(current,new)
        # lines = lines.lower()
        nf.write(lines)