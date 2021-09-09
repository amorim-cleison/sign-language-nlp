class FieldComposer:
    def __init__(self, fields, strategy):
        self.fields = fields
        self.strategy = strategy

    def compose_all_values(self, rows):
        """
        Example:
        `
        left_back           -                    -left_down_front     -                    -L                   -                    
        `
        """
        return list(
            map(
                lambda row: "-".join([
                    f"{(row[x]['value'] if row[x] else ''):<20}"
                    for x in self.fields
                ]), rows))

    def compose_as_words(self, rows):
        """
        Example:
        `
        lb--ldf--L-
        `
        """
        def compose_field(data):
            return ''.join([k[0] for k in str(data['value']).split('_')
                            ]) if data else ''

        return list(
            map(
                lambda row: "-".join(
                    [compose_field(row[f]) for f in self.fields]), rows))

    def compose_as_words_norm(self, rows):
        """
        Example:
        `
        l_b-___-ldf-___-L-
        `
        """
        def compose_field(field, data):
            values = str(data['value']) if data else ''

            if field.startswith("orientation") or field.startswith("movement"):
                values = values.split('_')
                return ''.join([("l" if "left" in values else
                                 "r" if "right" in values else "_"),
                                ("u" if "up" in values else
                                 "d" if "down" in values else "_"),
                                ("f" if "front" in values else
                                 "b" if "back" in values else "_")])
            else:
                return values

        return list(
            map(
                lambda row: "-".join(
                    [compose_field(f, row[f]) for f in self.fields]), rows))

    def compose_sep_feat(self, rows):
        """
        Example:
        `
        ['lb', '', 'ldf', '', 'L', '']
        `
        """
        def compose_field(data):
            return ''.join([k[0] for k in str(data['value']).split('_')
                            ]) if data else ''

        return list(
            map(lambda row: str([compose_field(row[f]) for f in self.fields]),
                rows))

    def run(self, rows):
        try:
            fn = self.FIELD_COMPOSE_STRATEGY[self.strategy]
            return fn(self, rows)
        except Exception as e:
            raise Exception(f"There was an error while running strategy "
                            f"'{self.strategy}' in FieldComposer: {repr(e)}")

    FIELD_COMPOSE_STRATEGY = {
        "all_values": compose_all_values,
        "as_words": compose_as_words,
        "as_words_norm": compose_as_words_norm,
        "as_sep_feat": compose_sep_feat,
    }
